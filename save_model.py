# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pfld import create_model
from generate_data import DataLoader

import time
# import matplotlib.pyplot as plt
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))

import tensorflow as tf
from tensorflow.python.framework import graph_util
import tfcoreml as tf_converter
import numpy as np
import argparse
import sys
# import matplotlib
# matplotlib.use('Agg')


def create_save_model(args, model_dir, graph, sess):
    # check data type
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print(i)   # i.name if you want just a name
    print("finish check")
    # save graphdef file to pb
    print("Save frozen graph")
    graphdef_n = "original_98_frozen.pb"
    graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ["pfld_inference/fc/BiasAdd"])
    tf.train.write_graph(graph_def, model_dir, graphdef_n, as_text=False)

    # save SavedModel
    print("get tensor")
    image_batch = graph.get_tensor_by_name('image_batch:0')
    landmarks_pre = graph.get_tensor_by_name('pfld_inference/fc/BiasAdd:0')
    print("start save saved_model")
    save_model_dir = os.path.join(model_dir, "SavedModel")
    # tf.saved_model.simple_save(sess, save_model_dir, inputs={"image_batch": image_batch}, outputs={"pfld_inference/fc/BiasAdd": landmarks_pre})
    builder = tf.saved_model.builder.SavedModelBuilder(save_model_dir)
    signature = tf.saved_model.predict_signature_def(
        {"image_batch": image_batch}, outputs={"pfld_inference/fc/BiasAdd": landmarks_pre}
    )

    # using custom tag instead of: tags=[tf.saved_model.tag_constants.SERVING]
    builder.add_meta_graph_and_variables(
        sess=sess, tags=[
            tf.saved_model.tag_constants.SERVING], signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    builder.save()
    print("finish save saved_model")

    # save tflite model
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/post_training_quantization.md
    # ref: https://github.com/seeouy/edgetpu_model_converter/blob/master/keras_to_edgetpu_model_converter.ipynb
    # https://www.gitmemory.com/issue/tensorflow/tensorflow/27880/513844787
    # https://www.tdi.co.jp/miso/tensorflow-tfrecord-02-datasetapi#TFRecordDataset_API

    converter = tf.lite.TFLiteConverter.from_saved_model(save_model_dir)

    # for aware int8 training
    converter.inference_type = tf.uint8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (0, 255)}  # mean, std_dev
    # relu6; x→min(max(0,x),6).
    converter.default_ranges_stats = (0,6)
    """
    # for post int8 quantization
    def representative_dataset_gen():
        test_loader = DataLoader(args.test_list, args, "test")
        test_images, _ = test_loader.gen_tfrecord()
        for i in range(200):
            yield [test_images[i: i + 1]]
    converter.representative_dataset = representative_dataset_gen
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    """
    tflite_model = converter.convert()

    with open(os.path.join(model_dir, "pfld_aware_qint8_growing.tflite"), 'wb') as f:
        f.write(tflite_model)

    print("finish save tflite")


def create_coreml_model(model_dir, args):
    # coreml変換
    tf_converter.convert(tf_model_path=os.path.join(model_dir, "original_98_frozen.pb"),
                        mlmodel_path=os.path.join(model_dir, 'pfld.mlmodel'),
                        input_name_shape_dict={'image_batch:0':[1,args.image_size,args.image_size,3]},
                        output_feature_names=['pfld_inference/fc/BiasAdd:0'],
                        add_custom_layers=True
                        )


def main(args):
    with tf.Graph().as_default() as inf_g:
        image_batch = tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3),
                                     name='image_batch')
        landmark_batch = tf.placeholder(tf.float32, shape=(
            None, args.num_labels * 2), name='landmark_batch')

        phase_train_placeholder = tf.constant(False, name='phase_train')
        landmarks_pre, _, _ = create_model(
            image_batch, landmark_batch, phase_train_placeholder, args)

        save_params = tf.trainable_variables()
        saver = tf.train.Saver(save_params, max_to_keep=None)
        # quantize
        if args.num_quant < 64:
            print("=====================================")
            print("quantize by: ", args.num_quant)
            """
            tf.contrib.quantize.experimental_create_eval_graph(
                input_graph=inf_g,
                weight_bits=args.num_quant,
                activation_bits=args.num_quant,
                symmetric=False,
                quant_delay=None,
                scope=None
            )
            """
            # ref: https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/quantize
            tf.contrib.quantize.create_eval_graph(input_graph=inf_g)
        else:
            print("no quantize, so float: ", args.num_quant)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        inf_sess = tf.Session(
            graph=inf_g,
            config=tf.ConfigProto(
                gpu_options=gpu_options,
                allow_soft_placement=False,
                log_device_placement=False))
        inf_sess.run(tf.global_variables_initializer())
        inf_sess.run(tf.local_variables_initializer())

        with inf_sess.as_default():
            print('Model directory: {}'.format(args.pretrained_model))
            ckpt = tf.train.get_checkpoint_state(args.pretrained_model)
            print('ckpt: {}'.format(ckpt))
            model_path = ckpt.model_checkpoint_path
            assert (ckpt and model_path)
            epoch_start = int(
                model_path[model_path.find('model.ckpt-') + 11:]) + 1
            print('Checkpoint file: {}'.format(model_path))
            saver.restore(inf_sess, model_path)

            create_save_model(args, args.model_dir, inf_g, inf_sess)

    create_coreml_model(args.model_dir, args)


def parse_arguments(argv):
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_list', type=str, default='data/test_data/list.txt')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--num_labels', type=int, default=98)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default='models1/model_test')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_epoch', type=str, default='10,20,30,40,200,500')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--level', type=str, default='L5')
    parser.add_argument('--save_image_example', action='store_false')
    parser.add_argument('--depth_multi', type=float, default=1)
    parser.add_argument('--num_quant', type=int, default=64)
    parser.add_argument('--is_augment', type=str2bool, default=False, help='Whether to augment')

    return parser.parse_args(argv)


if __name__ == '__main__':
    print(sys.argv)
    main(parse_arguments(sys.argv[1:]))

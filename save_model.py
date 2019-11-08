# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model2 import create_model
import time
# import matplotlib.pyplot as plt
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import argparse
import sys
# import matplotlib
# matplotlib.use('Agg')


def create_save_model(model_dir, graph, sess):
    # save graphdef file to pb	
    print("Save frozen graph")	
    graphdef_n = "original_98_frozen.pb"	
    graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ["pfld_inference/fc/BiasAdd"])	
    tf.train.write_graph(graph_def,model_dir,graphdef_n,as_text=False)

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
    builder.add_meta_graph_and_variables(sess=sess,
                                        tags=[tf.saved_model.tag_constants.SERVING],
                                        signature_def_map={
                                            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
                                            }
                                            )
    builder.save()
    print("finish save saved_model")


def main(args):
    debug = (args.debug == 'True')
    print("args: ", args)
    np.random.seed(args.seed)
    time.sleep(3)

    with tf.Graph().as_default() as inf_g:
        image_batch = tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3),
                                     name='image_batch')
        landmark_batch = tf.placeholder(tf.float32, shape=(None, args.num_labels*2), name='landmark_batch')

        phase_train_placeholder = tf.constant(False, name='phase_train')
        landmarks_pre, _, _ = create_model(image_batch, landmark_batch, phase_train_placeholder, args)
        
        save_params = tf.trainable_variables()
        saver = tf.train.Saver(save_params, max_to_keep=None)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        inf_sess = tf.Session(graph=inf_g, config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=False, log_device_placement=False))
        inf_sess.run(tf.global_variables_initializer())
        inf_sess.run(tf.local_variables_initializer())

        with inf_sess.as_default():
            print('Model directory: {}'.format(args.pretrained_model))
            ckpt = tf.train.get_checkpoint_state(args.pretrained_model)
            print('ckpt: {}'.format(ckpt))
            model_path = ckpt.model_checkpoint_path
            assert (ckpt and model_path)
            epoch_start = int(model_path[model_path.find('model.ckpt-') + 11:]) + 1
            print('Checkpoint file: {}'.format(model_path))
            saver.restore(inf_sess, model_path)

            create_save_model(args.model_dir, inf_g, inf_sess)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--debug', type=str, default='False')
    parser.add_argument('--depth_multi', type=int, default=1)
    return parser.parse_args(argv)


if __name__ == '__main__':
    print(sys.argv)
    main(parse_arguments(sys.argv[1:]))

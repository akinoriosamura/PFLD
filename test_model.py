from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pfld import create_model
from generate_data import DataLoader


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import cv2
import shutil
import time
import argparse
import sys
import os


def main(args):
    print("args: ", args)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    else:
        shutil.rmtree(args.out_dir)
        os.mkdir(args.out_dir)

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
            print("quantize by: ", args.num_quant)
            tf.contrib.quantize.experimental_create_eval_graph(
                input_graph=inf_g,
                weight_bits=args.num_quant,
                activation_bits=args.num_quant,
                symmetric=False,
                quant_delay=None,
                scope=None
            )
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

            dataloader = DataLoader(args.test_list, args, "test")
            file_list, train_landmarks, train_attributes, euler_angles = dataloader.gen_data(
                args.test_list, args.num_labels)
            print(file_list)
            for file in file_list:
                filename = os.path.split(file)[-1]
                image = cv2.imread(file)
                # image = cv2.resize(image, (image_size, image_size))
                input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                input = cv2.resize(input, (args.image_size, args.image_size))
                input = input.astype(np.float32) / 256.0
                input = np.expand_dims(input, 0)
                # print(input.shape)

                feed_dict = {
                    image_batch: input
                }
                st = time.time()
                pre_landmarks = inf_sess.run(landmarks_pre, feed_dict=feed_dict)
                # print(pre_landmarks)
                print("elaps: ", time.time() - st)
                pre_landmark = pre_landmarks[0]

                h, w, _ = image.shape
                pre_landmark = pre_landmark.reshape(-1, 2) * [h, w]
                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(image, (x, y), 1, (0, 0, 255))
                print(os.path.join(args.out_dir, filename))
                cv2.imwrite(os.path.join(args.out_dir, filename), image)

def parse_arguments(argv):
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--test_list', type=str, default='data/test_data/list.txt')
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--num_labels', type=int, default=98)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_epoch', type=str, default='10,20,30,40,200,500')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--level', type=str, default='L5')
    parser.add_argument('--save_image_example', action='store_false')
    parser.add_argument('--depth_multi', type=float, default=1)
    parser.add_argument('--out_dir', type=str, default='sample_result')
    parser.add_argument('--num_quant', type=int, default=64)
    parser.add_argument('--is_augment', type=str2bool, default=False, help='Whether to augment')

    return parser.parse_args(argv)


if __name__ == '__main__':
    print(sys.argv)
    main(parse_arguments(sys.argv[1:]))

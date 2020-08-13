# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import LandmarkImage, LandmarkImage_98

import time


def conv2d(net, stride, channel, kernel, depth, scope):
    num_channel = depth(channel)
    net = slim.conv2d(net, num_channel, [kernel, kernel], stride=stride, scope=scope)

    return net

def GAP(net, scope):
    # Global average pool
    n = int(net.get_shape().dims[1])
    net = slim.avg_pool2d(net, [1, n], stride=[1, n], scope=scope)
    net = slim.flatten(net)

    return net

def HeatMap(output, _in):
    # 1*136 / transform / 112*112*1
    back = tf.constant(0, shape=[_in.get_shape().dims[1], _in.get_shape().dims[2]])
    _output = tf.cast(tf.reshape(output, [-1, 2]), dtype=tf.int32)
    _pixels = tf.constant([(x, y) for y in range(self.img_size) for x in range(self.img_size)],
                                      dtype=tf.float32,shape=[1,self.img_size,self.img_size,2])
    # x = tf.constant([[1, 1, 1], [1, 1, 1]])
    # tf.reduce_sum(x, 0)  # [2, 2, 2]

    tmp = tf.constant(0, shape=[_in.get_shape().dims[1], _in.get_shape().dims[2], 1])
    return tmp


def XinNingNetwork(input, weight_decay, batch_norm_params, num_labels, depth_multi, min_depth=8):
    print("labels; ", num_labels)
    time.sleep(3)

    def depth(d):
        return max(int(d * depth_multi), min_depth)

    with tf.variable_scope('pfld_inference'):
        features = {}
        # normalizer_fn=slim.batch_norm,
        with slim.arg_scope(
            [slim.conv2d],
            activation_fn=tf.nn.relu6,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            biases_initializer=tf.zeros_initializer(),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params,
            padding='SAME'
            ):
            print('PFLD input shape({}): {}'.format(input.name, input.get_shape()))
            # 112*112*3(1) / conv3*3 / c:16,n:1,s:2
            net = conv2d(input, stride=2, channel=16, kernel=3, depth=depth, scope='conv1_1')
            print(net.name, net.get_shape())
            # 56*56*16 / conv3*3 / c:32,n:1,s:2
            net = conv2d(net, stride=2, channel=32, kernel=3, depth=depth, scope='conv1_2')
            print(net.name, net.get_shape())
            # 28*28*32 / pool2*2 / c:32,n:1,s:2
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1_2', padding='SAME')
            print(net.name, net.get_shape())
            # 14*14*32 / conv3*3 / c:64,n:1,s:2
            net = conv2d(net, stride=2, channel=64, kernel=3, depth=depth, scope='conv1_2.1')
            print(net.name, net.get_shape())
            # 7*7*64 / global_pool / c:64,n:1
            net = GAP(net, scope='pool1_2.1')
            print(net.name, net.get_shape())
            # 14*14*32 / conv3*3 / c:64,n:1,s:2
            net = conv2d(net, stride=2, channel=64, kernel=3, depth=depth, scope='conv1_3')
            print(net.name, net.get_shape())
            # 7*7*64 / pool2*2 / c:64,n:1,s:2
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1_3', padding='SAME')
            print(net.name, net.get_shape())
            # 4*4*64 / conv3*3 / c:64,n:1,s:2
            net = conv2d(net, stride=2, channel=64, kernel=3, depth=depth, scope='conv1_3.1')
            print(net.name, net.get_shape())
            # 2*2*64 / global_pool / c:64,n:1
            net = GAP(net, scope='pool1_3.1')
            print(net.name, net.get_shape())
            # 4*4*64 / conv3*3 / c:64,n:1,s:2
            net = conv2d(net, stride=2, channel=64, kernel=3, depth=depth, scope='conv1_4')
            print(net.name, net.get_shape())
            # 2*2*64 / global_pool / c:64,n:1
            net = GAP(net, scope='pool1_4.1')
            print(net.name, net.get_shape())
            # 1*1*64*3 / concat / 1*1*192
            flattened_1 = slim.flatten(net)
            print(flattened_1.name, flattened_1.get_shape())
            # 1*1*192 / fc / 1*136
            output_1 = slim.fully_connected(flattened_1, num_outputs=num_labels*2, scope='fc_1')
            print(output_1.name, output_1.get_shape())
            # 1*136 / transform / 112*112*1
            heatmap = HeatMap(output_1, input)
            print(heatmap.name, heatmap.get_shape())

            # 112*112*1*2 / concat / 112*112*2
            concatted = tf.concat([input, heatmap], 1)
            print(concatted.name, concatted.get_shape())
            # 112*112*2 / conv3*3 / c:8,n:1,s:2
            net = conv2d(concatted, stride=2, channel=8, kernel=3, depth=depth, scope='conv2_1')
            print(net.name, net.get_shape())
            # 56*56*8 / pool3*3 / c:28,n:1,s:2
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2_1', padding='SAME')
            print(net.name, net.get_shape())
            # 28*28*8 / conv3*3 / c:16,n:1,s:1
            net = conv2d(net, stride=1, channel=16, kernel=3, depth=depth, scope='conv2_2')
            print(net.name, net.get_shape())
            # 28*28*16 / pool3*3 / c:16,n:1,s:2
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2_2', padding='SAME')
            print(net.name, net.get_shape())
            # 14*14*16 / conv3*3 / c:64,n:1,s:1
            net = conv2d(net, stride=1, channel=64, kernel=3, depth=depth, scope='conv2_2.1')
            print(net.name, net.get_shape())
            # 14*14*64 / global_pool / c:64,n:1
            net = GAP(net, scope='pool2_2.1')
            print(net.name, net.get_shape())
            # 14*14*16 / conv3*3 / c:64,n:1,s:2
            net = conv2d(net, stride=2, channel=64, kernel=3, depth=depth, scope='conv2_3')
            print(net.name, net.get_shape())
            # 7*7*64 / pool3*3 / c:64,n:1,s:2
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2_3', padding='SAME')
            print(net.name, net.get_shape())
            # 3*3*64 / conv3*3 / c:64,n:1,s:1
            net = conv2d(net, stride=2, channel=64, kernel=3, depth=depth, scope='conv2_3.1')
            print(net.name, net.get_shape())
            # 3*3*64 / global_pool / c:64,n:1
            net = GAP(net, scope='pool2_3.1')
            print(net.name, net.get_shape())
            # 3*3*64 / conv3*3 / c:64,n:1,s:2
            net = conv2d(net, stride=2, channel=64, kernel=3, depth=depth, scope='conv2_4')
            print(net.name, net.get_shape())
            # 2*2*64 / global_pool / c:64,n:1
            net = GAP(net, scope='pool2_4.1')
            print(net.name, net.get_shape())
            # 1*1*64*3 / concat / 1*1*192
            flattened_2 = slim.flatten(net)
            print(flattened_2.name, flattened_2.get_shape())
            # 1*1*192 / fc / 1*136
            output_2 = slim.fully_connected(flattened_2, num_outputs=num_labels*2, scope='fc_2')
            print("last layer name")
            print(output_2.name, output_2.get_shape())
            return output_2


def create_model(input, landmark, phase_train, args):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,  # tf.GraphKeys.UPDATE_OPS,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        'is_training': phase_train
    }

    landmark_dim = int(landmark.get_shape()[-1])
    print("labels; ", args.num_labels)
    time.sleep(3)
    landmarks_pre = XinNingNetwork(input, args.weight_decay, batch_norm_params, args.num_labels, args.depth_multi)
    # loss
    landmarks_loss = tf.reduce_sum(tf.square(landmarks_pre - landmark), axis=1)
    landmarks_loss = tf.reduce_mean(landmarks_loss)

    # euler_angles_pre = pfld_auxiliary(features, args.weight_decay, batch_norm_params)

    print("==========finish define graph===========")

    # return landmarks_loss, landmarks, heatmap_loss, HeatMaps
    return landmarks_pre, landmarks_loss# , euler_angles_pre


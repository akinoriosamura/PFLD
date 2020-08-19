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

def label_heatmap(land):
    # land: [2]
    x_range = tf.range(0, 112)
    y_range = tf.range(0, 112)
    xx, yy = tf.meshgrid(x_range, y_range)
    land_x = tf.math.multiply(land[0], 112)
    land_y = tf.math.multiply(land[1], 112)
    _xx = tf.pow(tf.subtract(tf.cast(xx, dtype=tf.float32), land_x), 2)
    _yy = tf.pow(tf.subtract(tf.cast(yy, dtype=tf.float32), land_y), 2)
    d2 = tf.add(_xx, _yy)
    heatmap_tmp = tf.math.exp(tf.math.negative(d2), name="one_label_heatmap")

    return heatmap_tmp

def img_heatmap(land):
    # land: [68, 2]
    one_label_heatmap = tf.map_fn(label_heatmap, land)
    labels_heatmap = tf.reduce_sum(one_label_heatmap, 0, name="label_heatmap")

    return labels_heatmap

def HeatMap(output, _in, num_labels):
    # ?*136 / transform / ?*112*112*1
    # import pdb;pdb.set_trace()
    im_dim = _in.get_shape().dims[1].value
    # landmark: [?, 68, 2]
    landmark = tf.reshape(output, [-1, 68, 2])
    heatmap = tf.map_fn(img_heatmap, landmark)
    heatmap = tf.expand_dims(heatmap, -1, name="heatmap")
    print(heatmap.name, heatmap.get_shape())

    return heatmap, [landmark, heatmap]

def _HeatMap(output, _in, num_labels):
    # ?*136 / transform / ?*112*112*1
    heatmap = _in[:,:,:,0]
    heatmap = tf.expand_dims(heatmap, -1, name="heatmap")
    print(heatmap.name, heatmap.get_shape())

    return heatmap, []

def XinNingNetwork1(input, is_training, weight_decay, batch_norm_params, num_labels, depth_multi, min_depth=8):
    print("labels; ", num_labels)
    time.sleep(3)

    def depth(d):
        return max(int(d * depth_multi), min_depth)

    with tf.variable_scope('pfld_inference1'):
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
            padding='SAME',
            # trainable=is_training
            ):
            print('PFLD input shape({}): {}'.format(input.name, input.get_shape()))
            # 112*112*3(1) / conv3*3 / c:16,n:1,s:2
            conv1_1 = conv2d(input, stride=2, channel=16, kernel=3, depth=depth, scope='conv1_1')
            print(conv1_1.name, conv1_1.get_shape())
            # 56*56*16 / conv3*3 / c:32,n:1,s:2
            conv1_2 = conv2d(conv1_1, stride=2, channel=32, kernel=3, depth=depth, scope='conv1_2')
            print(conv1_2.name, conv1_2.get_shape())
            # 28*28*32 / pool2*2 / c:32,n:1,s:2
            pool1_2 = slim.max_pool2d(conv1_2, kernel_size=[2, 2], stride=2, scope='pool1_2', padding='SAME')
            print(pool1_2.name, pool1_2.get_shape())
            # 14*14*32 / conv3*3 / c:64,n:1,s:2
            conv1_2_1 = conv2d(pool1_2, stride=2, channel=64, kernel=3, depth=depth, scope='conv1_2.1')
            print(conv1_2_1.name, conv1_2_1.get_shape())
            # 7*7*64 / global_pool / c:64,n:1
            pool1_2_1 = slim.avg_pool2d(conv1_2_1, [7, 7], stride=[7, 7], scope='pool1_2.1', padding='SAME')
            print(pool1_2_1.name, pool1_2_1.get_shape())
            # 14*14*32 / conv3*3 / c:64,n:1,s:2
            conv1_3 = conv2d(pool1_2, stride=2, channel=64, kernel=3, depth=depth, scope='conv1_3')
            print(conv1_3.name, conv1_3.get_shape())
            # 7*7*64 / pool2*2 / c:64,n:1,s:2
            pool1_3 = slim.max_pool2d(conv1_3, kernel_size=[2, 2], stride=2, scope='pool1_3', padding='SAME')
            print(pool1_3.name, pool1_3.get_shape())
            # 4*4*64 / conv3*3 / c:64,n:1,s:2
            conv1_3_1 = conv2d(pool1_3, stride=2, channel=64, kernel=3, depth=depth, scope='conv1_3.1')
            print(conv1_3_1.name, conv1_3_1.get_shape())
            # 2*2*64 / global_pool / c:64,n:1
            pool1_3_1 = slim.avg_pool2d(conv1_3_1, [2, 2], stride=[2, 2], scope='pool1_3.1', padding='SAME')
            print(pool1_3_1.name, pool1_3_1.get_shape())
            # 4*4*64 / conv3*3 / c:64,n:1,s:2
            conv1_4 = conv2d(pool1_3, stride=2, channel=64, kernel=3, depth=depth, scope='conv1_4')
            print(conv1_4.name, conv1_4.get_shape())
            # 2*2*64 / global_pool / c:64,n:1
            pool1_4_1 = slim.avg_pool2d(conv1_4, [2, 2], stride=[2, 2], scope='pool1_4.1', padding='SAME')
            print(pool1_4_1.name, pool1_4_1.get_shape())
            # 1*1*64*3() / concat / 1*1*192
            concatted_1 = tf.concat([pool1_2_1, pool1_3_1, pool1_4_1], 3)
            print(concatted_1.name, concatted_1.get_shape())
            flattened_1 = slim.flatten(concatted_1)
            print(flattened_1.name, flattened_1.get_shape())
            # 1*1*192 / fc / 1*136
            output_1 = slim.fully_connected(flattened_1, num_outputs=num_labels*2, scope='fc_1')
            print(output_1.name, output_1.get_shape())
            # 1*136 / transform / 112*112*1
            heatmap, _heat_values = _HeatMap(output_1, input, num_labels)
            print(heatmap.name, heatmap.get_shape())
            print("=== finish stage 1 ===")

            return output_1, heatmap, _heat_values


def XinNingNetwork2(input, heatmap, is_training, weight_decay, batch_norm_params, num_labels, depth_multi, min_depth=8):
    print("labels; ", num_labels)
    time.sleep(3)

    def depth(d):
        return max(int(d * depth_multi), min_depth)

    with tf.variable_scope('pfld_inference2'):
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
            padding='SAME',
            # trainable=is_training
            ):
            print('PFLD input shape({}): {}'.format(input.name, input.get_shape()))
            print("=== start stage 2 ===")
            print(heatmap.name, heatmap.get_shape())
            # 112*112*1*2 / concat / 112*112*2
            concatted_2 = tf.concat([input, heatmap], 3)
            print(concatted_2.name, concatted_2.get_shape())
            # 112*112*2 / conv3*3 / c:8,n:1,s:2
            conv2_1 = conv2d(concatted_2, stride=2, channel=8, kernel=3, depth=depth, scope='conv2_1')
            print(conv2_1.name, conv2_1.get_shape())
            # 56*56*8 / pool3*3 / c:28,n:1,s:2
            pool2_1 = slim.max_pool2d(conv2_1, kernel_size=[3, 3], stride=2, scope='pool2_1', padding='SAME')
            print(pool2_1.name, pool2_1.get_shape())
            # 28*28*8 / conv3*3 / c:16,n:1,s:1
            conv2_2 = conv2d(pool2_1, stride=1, channel=16, kernel=3, depth=depth, scope='conv2_2')
            print(conv2_2.name, conv2_2.get_shape())
            # 28*28*16 / pool3*3 / c:16,n:1,s:2
            pool2_2 = slim.max_pool2d(conv2_2, kernel_size=[3, 3], stride=2, scope='pool2_2', padding='SAME')
            print(pool2_2.name, pool2_2.get_shape())
            # 14*14*16 / conv3*3 / c:64,n:1,s:1
            conv2_2_1 = conv2d(pool2_2, stride=1, channel=64, kernel=3, depth=depth, scope='conv2_2.1')
            print(conv2_2_1.name, conv2_2_1.get_shape())
            # 14*14*64 / global_pool / c:64,n:1
            pool2_2_1 = slim.avg_pool2d(conv2_2_1, [14, 14], stride=[14, 14], scope='pool2_2.1', padding='SAME')
            print(pool2_2_1.name, pool2_2_1.get_shape())
            # 14*14*16 / conv3*3 / c:64,n:1,s:2
            conv2_3 = conv2d(pool2_2, stride=2, channel=64, kernel=3, depth=depth, scope='conv2_3')
            print(conv2_3.name, conv2_3.get_shape())
            # 7*7*64 / pool3*3 / c:64,n:1,s:2
            pool2_3 = slim.max_pool2d(conv2_3, kernel_size=[3, 3], stride=2, scope='pool2_3', padding='SAME')
            print(pool2_3.name, pool2_3.get_shape())
            # 3*3*64 / conv3*3 / c:64,n:1,s:1
            conv2_3_1 = conv2d(pool2_3, stride=2, channel=64, kernel=3, depth=depth, scope='conv2_3.1')
            print(conv2_3_1.name, conv2_3_1.get_shape())
            # 3*3*64 / global_pool / c:64,n:1
            pool2_3_1 = slim.avg_pool2d(conv2_3_1, [3, 3], stride=[3, 3], scope='pool2_3.1', padding='SAME')
            print(pool2_3_1.name, pool2_3_1.get_shape())
            # 3*3*64 / conv3*3 / c:64,n:1,s:2
            conv2_4 = conv2d(pool2_3, stride=2, channel=64, kernel=3, depth=depth, scope='conv2_4')
            print(conv2_4.name, conv2_4.get_shape())
            # 2*2*64 / global_pool / c:64,n:1
            pool2_4_1 = slim.avg_pool2d(conv2_4, [2, 2], stride=[2, 2], scope='pool2_4.1', padding='SAME')
            print(pool2_4_1.name, pool2_4_1.get_shape())
            # 1*1*64*3 / concat / 1*1*192
            concatted_2 = tf.concat([pool2_2_1, pool2_3_1, pool2_4_1], 3)
            print(concatted_2.name, concatted_2.get_shape())
            flattened = slim.flatten(concatted_2)
            print(flattened.name, flattened.get_shape())
            # 1*1*192 / fc / 1*136
            output_2 = slim.fully_connected(flattened, num_outputs=num_labels*2, scope='fc_2')
            print("last layer name")
            print(output_2.name, output_2.get_shape())

            return output_2


def create_model(input, landmark, phase_train, args, mean_shape=None):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,  # tf.GraphKeys.UPDATE_OPS,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        'is_training': phase_train,
        'trainable': phase_train
    }
    # trainableはbatch_normの内部にあるvariablesをGraphKeys.TRAINABLE_VARIABLESに登録するかどうかを制御するためのboolパラメーターなのに対して、
    # is_trainingはmoving_meanやmoving_varianceの挙動に関するboolパラメーターです。どちらもデフォルトでTrueですが、学習以外の時は明示的にFalseを設定する必要があります。
    # 特にis_trainingがTrueのままの場合、同じ入力に対してbatch_normが毎回違う出力をしてしまい、モデルの再現性がなくなる場合があるので注意が必要です。

    landmark_dim = int(landmark.get_shape()[-1])
    print("labels; ", args.num_labels)
    time.sleep(3)
    stage1_training = True
    # if not stage1_training:
    #     batch_norm_params['is_training'] = stage1_training
    stage2_training = True
    _, heatmap, _heat_values = XinNingNetwork1(input, stage1_training, args.weight_decay, batch_norm_params, args.num_labels, args.depth_multi)
    landmarks_out = XinNingNetwork2(input, heatmap, stage2_training, args.weight_decay, batch_norm_params, args.num_labels, args.depth_multi)
    # loss
    # landmarks_pre = tf.map_fn(lambda x: tf.add(x, tf.cast(tf.constant(mean_shape), dtype=tf.float32)), landmarks_out)
    landmarks_pre = landmarks_out
    _heat_values.extend([landmarks_out])

    # euler_angles_pre = pfld_auxiliary(features, args.weight_decay, batch_norm_params)

    print("==========finish define graph===========")

    # return landmarks_loss, landmarks, heatmap_loss, HeatMaps
    return landmarks_pre, _heat_values# , euler_angles_pre


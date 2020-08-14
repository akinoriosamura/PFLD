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


def HeatMap(output, _in, num_labels):
    # 1*136 / transform / 112*112*1
    # back = tf.constant(0, shape=[_in.get_shape().dims[1], _in.get_shape().dims[2]])
    # _output = tf.cast(tf.reshape(output, [-1, 2]), dtype=tf.int32)
    # _pixels = tf.constant([(x, y) for y in range(self.img_size) for x in range(self.img_size)],
    #                                   dtype=tf.float32,shape=[1,self.img_size,self.img_size,2])
    # x = tf.constant([[1, 1, 1], [1, 1, 1]])
    # tf.reduce_sum(x, 0)  # [2, 2, 2]

    # ref: https://stackoverflow.com/questions/57261091/tensorflow-landmark-heatmap
    im_dim = _in.get_shape().dims[1].value
    heatmap_template = tf.constant(0.6, shape=[4, 4])
    print(heatmap_template.name, heatmap_template.get_shape())
    # _output = tf.squeeze(output)
    # import pdb;pdb.set_trace()
    print(output.name, output.get_shape())
    locations = tf.cast(tf.multiply(tf.reshape(output, [-1, 68, 2]), [im_dim, im_dim]), dtype=tf.int32)
    # locations = tf.random.uniform((num_labels, 2), maxval=im_dim, dtype=tf.int32)
    print(locations.name, locations.get_shape())
    # import pdb;pdb.set_trace()
    # scat_updates = tf.Variable(1, shape=[-1, num_labels, 112])
    _scat_updates = tf.matmul(_in[:,:68,:,0], tf.zeros([112,112], tf.float32))
    scat_updates = tf.add(_scat_updates, tf.ones([68,112], tf.float32))
    print(scat_updates.name, scat_updates.get_shape())
    # scat_shape = tf.constant([-1, 112, 112])
    scat_shape = tf.matmul(_in[:,:,:,0], tf.zeros([112,112], tf.float32))
    centers = tf.cast(tf.scatter_nd(locations, scat_updates, tf.shape(_in[:,:,:,0])), dtype=tf.float32)
    print(centers.name, centers.get_shape())
    print(centers[None, :, :, :, None].get_shape())
    print(centers)
    heatmap  = tf.nn.conv2d(centers[:, :, :, None], heatmap_template[:, :, None, None], (1, 1, 1, 1), 'SAME')
    # _heatmap  = slim.conv2d(centers[:, :, :, None], None, [8, 8], stride=1, scope="merge_heatmap")
    print(heatmap.name, heatmap.get_shape())
    #heatmap = _heatmap[:, :, :, 0]
    #print(heatmap.name, heatmap.get_shape())
    #heatmap = tf.expand_dims(heatmap, -1)

    # tmp = tf.constant(0, shape=[_in.get_shape().dims[0], _in.get_shape().dims[1], _in.get_shape().dims[2], 1])
    return heatmap, [locations, _scat_updates, scat_updates, scat_shape, centers]


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
            conv1_1 = conv2d(input, stride=2, channel=16, kernel=3, depth=depth, scope='conv1_1')
            print(conv1_1.name, conv1_1.get_shape())
            # 56*56*16 / conv3*3 / c:32,n:1,s:2
            conv1_2 = conv2d(conv1_1, stride=2, channel=32, kernel=3, depth=depth, scope='conv1_2')
            print(conv1_2.name, conv1_2.get_shape())
            # 28*28*32 / pool2*2 / c:32,n:1,s:2
            pool1_2 = slim.max_pool2d(conv1_2, kernel_size=[2, 2], stride=2, scope='pool1_2', padding='SAME')
            print(pool1_2.name, pool1_2.get_shape())
            # 14*14*32 / conv3*3 / c:64,n:1,s:2
            conv1_2_1 = conv2d(pool1_2, stride=2, channel=512, kernel=3, depth=depth, scope='conv1_2.1')
            print(conv1_2_1.name, conv1_2_1.get_shape())
            # 7*7*64 / global_pool / c:64,n:1
            pool1_2_1 = slim.avg_pool2d(conv1_2_1, [7, 7], stride=[7, 7], scope='pool1_2.1', padding='SAME')
            print(pool1_2_1.name, pool1_2_1.get_shape())
            # 14*14*32 / conv3*3 / c:64,n:1,s:2
            conv1_3 = conv2d(pool1_2, stride=2, channel=512, kernel=3, depth=depth, scope='conv1_3')
            print(conv1_3.name, conv1_3.get_shape())
            # 7*7*64 / pool2*2 / c:64,n:1,s:2
            pool1_3 = slim.max_pool2d(conv1_3, kernel_size=[2, 2], stride=2, scope='pool1_3', padding='SAME')
            print(pool1_3.name, pool1_3.get_shape())
            # 4*4*64 / conv3*3 / c:64,n:1,s:2
            conv1_3_1 = conv2d(pool1_3, stride=2, channel=512, kernel=3, depth=depth, scope='conv1_3.1')
            print(conv1_3_1.name, conv1_3_1.get_shape())
            # 2*2*64 / global_pool / c:64,n:1
            pool1_3_1 = slim.avg_pool2d(conv1_3_1, [2, 2], stride=[2, 2], scope='pool1_3.1', padding='SAME')
            print(pool1_3_1.name, pool1_3_1.get_shape())
            # 4*4*64 / conv3*3 / c:64,n:1,s:2
            conv1_4 = conv2d(pool1_3, stride=2, channel=512, kernel=3, depth=depth, scope='conv1_4')
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
            heatmap, _heat_values = HeatMap(output_1, input, num_labels)
            print(heatmap.name, heatmap.get_shape())
            print("=== finish stage 1 ===")

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
            conv2_2_1 = conv2d(pool2_2, stride=1, channel=512, kernel=3, depth=depth, scope='conv2_2.1')
            print(conv2_2_1.name, conv2_2_1.get_shape())
            # 14*14*64 / global_pool / c:64,n:1
            pool2_2_1 = slim.avg_pool2d(conv1_4, [14, 14], stride=[14, 14], scope='pool2_2.1', padding='SAME')
            print(pool2_2_1.name, pool2_2_1.get_shape())
            # 14*14*16 / conv3*3 / c:64,n:1,s:2
            conv2_3 = conv2d(pool2_2, stride=2, channel=512, kernel=3, depth=depth, scope='conv2_3')
            print(conv2_3.name, conv2_3.get_shape())
            # 7*7*64 / pool3*3 / c:64,n:1,s:2
            pool2_3 = slim.max_pool2d(conv2_3, kernel_size=[3, 3], stride=2, scope='pool2_3', padding='SAME')
            print(pool2_3.name, pool2_3.get_shape())
            # 3*3*64 / conv3*3 / c:64,n:1,s:1
            conv2_3_1 = conv2d(pool2_3, stride=2, channel=512, kernel=3, depth=depth, scope='conv2_3.1')
            print(conv2_3_1.name, conv2_3_1.get_shape())
            # 3*3*64 / global_pool / c:64,n:1
            pool2_3_1 = slim.avg_pool2d(conv2_3_1, [3, 3], stride=[3, 3], scope='pool2_3.1', padding='SAME')
            print(pool2_3_1.name, pool2_3_1.get_shape())
            # 3*3*64 / conv3*3 / c:64,n:1,s:2
            conv2_4 = conv2d(pool2_3, stride=2, channel=512, kernel=3, depth=depth, scope='conv2_4')
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

            return output_1, heatmap, _heat_values


def create_model(input, landmark, phase_train, args, mean_shape):
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
    landmarks_out, heatmap, _heat_values = XinNingNetwork(input, args.weight_decay, batch_norm_params, args.num_labels, args.depth_multi)
    # loss
    landmarks_pre = tf.map_fn(lambda x: tf.add(x, tf.cast(tf.constant(mean_shape), dtype=tf.float32)), landmarks_out)
    landmarks_loss = tf.reduce_sum(tf.square(landmarks_pre - landmark), axis=1)
    landmarks_loss = tf.reduce_mean(landmarks_loss)
    _heat_values.extend([landmarks_out, landmarks_pre])

    # euler_angles_pre = pfld_auxiliary(features, args.weight_decay, batch_norm_params)

    print("==========finish define graph===========")

    # return landmarks_loss, landmarks, heatmap_loss, HeatMaps
    return landmarks_pre, landmarks_loss, heatmap, _heat_values# , euler_angles_pre


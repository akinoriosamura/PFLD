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

    print(net.name, net.get_shape())

    return net


def invertedbottleneck(net, stride, up_sample, channel, depth, scope):
    prev_output = net
    net = slim.conv2d(
        net,
        up_sample * net.get_shape().as_list()[-1],
        [1, 1],
        scope=scope + '/conv2d_1'
        )
    print(net.name, net.get_shape())
    net = slim.separable_conv2d(net, None, [3, 3],
                                depth_multiplier=1,
                                stride=stride,
                                scope=scope + '/separable2d')
    print(net.name, net.get_shape())
    num_channel = depth(channel)
    net = slim.conv2d(
        net,
        num_channel,
        [1, 1],
        activation_fn=None,
        scope=scope + '/conv2d_2'
        )
    print(net.name, net.get_shape())

    if stride == 1:
        if prev_output.get_shape().as_list(
        )[-1] != net.get_shape().as_list()[-1]:
            # Assumption based on previous ResNet papers: If the number of filters doesn't match,
            # there should be a conv 1x1 operation.
            # reference(pytorch) :
            # https://github.com/MG2033/MobileNet-V2/blob/master/layers.py#L29
            prev_output = slim.conv2d(
                prev_output,
                num_channel,
                [1, 1],
                activation_fn=None,
                biases_initializer=None,
                scope=scope + '/conv2d_3'
                )
            print(net.name, net.get_shape())

        # as described in Figure 4.
        net = tf.add(prev_output, net, name=scope + '/add')
        print(net.name, net.get_shape())
    return net


def pfld_backbone(input, weight_decay, batch_norm_params, num_labels, depth_multi, min_depth=1):
    print("labels; ", num_labels)
    time.sleep(3)

    def depth(d):
        return max(int(d * depth_multi), min_depth)

    with tf.variable_scope('pfld_inference'):
        features = {}
        # normalizer_fn=slim.batch_norm,
        with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d],
            activation_fn=tf.nn.relu6,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            biases_initializer=tf.zeros_initializer(),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params,
            padding='SAME'
            ):
            print('PFLD input shape({}): {}'.format(input.name, input.get_shape()))
            # 112*112*3 / conv3*3 / c:64,n:1,s:2
            conv1 = conv2d(input, stride=2, channel=64, kernel=3, depth=depth, scope='conv1')
            # 56*56*64 / depthwiseconv3*3 / c:64,n:1,s:1
            conv2 = slim.separable_conv2d(conv1, depth(64), [3, 3], depth_multiplier=1, stride=1, scope='conv2/dwise')
            print(conv2.name, conv2.get_shape())
            # 56*56*64 / InverseBottleneck / up_s:2,c:64,n:5,s:2
            conv3_1 = invertedbottleneck(conv2, stride=2, up_sample=2, channel=64, depth=depth, scope='conv3_1/inbottleneck')
            conv3_2 = invertedbottleneck(conv3_1, stride=1, up_sample=2, channel=64, depth=depth, scope='conv3_2/inbottleneck')
            conv3_3 = invertedbottleneck(conv3_2, stride=1, up_sample=2, channel=64, depth=depth, scope='conv3_3/inbottleneck')
            conv3_4 = invertedbottleneck(conv3_3, stride=1, up_sample=2, channel=64, depth=depth, scope='conv3_4/inbottleneck')
            conv3_5 = invertedbottleneck(conv3_4, stride=1, up_sample=2, channel=64, depth=depth, scope='conv3_5/inbottleneck')
            features['auxiliary_input'] = conv3_5
            # 28*28*64 / InverseBottleneck / up_s:2,c:128,n:1,s:2
            conv4 = invertedbottleneck(conv3_5, stride=2, up_sample=2, channel=128, depth=depth, scope='conv4/inbottleneck')
            # 14*14*128 / InverseBottleneck / up_s:4,c:128,n:6,s:1
            conv5_1 = invertedbottleneck(conv4, stride=1, up_sample=4, channel=128, depth=depth, scope='conv5_1/inbottleneck')
            conv5_2 = invertedbottleneck(conv5_1, stride=1, up_sample=4, channel=128, depth=depth, scope='conv5_2/inbottleneck')
            conv5_3 = invertedbottleneck(conv5_2, stride=1, up_sample=4, channel=128, depth=depth, scope='conv5_3/inbottleneck')
            conv5_4 = invertedbottleneck(conv5_3, stride=1, up_sample=4, channel=128, depth=depth, scope='conv5_4/inbottleneck')
            conv5_5 = invertedbottleneck(conv5_4, stride=1, up_sample=4, channel=128, depth=depth, scope='conv5_5/inbottleneck')
            conv5_6 = invertedbottleneck(conv5_5, stride=1, up_sample=4, channel=128, depth=depth, scope='conv5_6/inbottleneck')
            # 14*14*128 / InverseBottleneck / up_s:2,c:16,n:1,s:1
            conv6 = invertedbottleneck(conv5_6, stride=1, up_sample=2, channel=16, depth=depth, scope='conv6/inbottleneck')
            # 14*14*16 / conv3*3 / c:32,n:1,s:2
            conv7 = conv2d(conv6, stride=2, channel=depth(32), kernel=3, depth=depth, scope='conv7')
            # 7*7*32 / conv7*7 / c:128,n:1,s:1
            # conv8 = slim.conv2d(conv7, depth(128), [7, 7], stride=1, padding='VALID', scope='conv8')
            # for img size84
            conv8 = slim.conv2d(conv7, depth(128), [5, 5], stride=1, padding='VALID', scope='conv8')
            print(conv8.name, conv8.get_shape())
            avg_pool1 = slim.avg_pool2d(conv6, [conv6.get_shape()[1], conv6.get_shape()[2]], stride=1)
            print(avg_pool1.name, avg_pool1.get_shape())
            avg_pool2 = slim.avg_pool2d(conv7, [conv7.get_shape()[1], conv7.get_shape()[2]], stride=1)
            print(avg_pool2.name, avg_pool2.get_shape())
            # pfld_inference/AvgPool2D_1/AvgPool:0

            s1 = slim.flatten(avg_pool1)
            s2 = slim.flatten(avg_pool2)
            # 1*1*128
            s3 = slim.flatten(conv8)
            multi_scale = tf.concat([s1, s2, s3], 1)
            landmarks = slim.fully_connected(multi_scale, num_outputs=num_labels*2, activation_fn=None, scope='fc')
            print("last layer name")
            print(landmarks.name, landmarks.get_shape())
            return features, landmarks


def pfld_auxiliary(features, weight_decay, batch_norm_params):
    # add the auxiliary net
    # : finish the loss function
    print('\nauxiliary net')
    with slim.arg_scope([slim.convolution2d, slim.fully_connected], \
                        activation_fn=tf.nn.relu,\
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        pfld_input = features['auxiliary_input']
        net_aux = slim.convolution2d(pfld_input, 128, [3, 3], stride=2, scope='pfld_conv1')
        print(net_aux.name, net_aux.get_shape())
        # net = slim.max_pool2d(net, kernel_size=[3, 3], stride=1, scope='pool1', padding='SAME')
        net_aux = slim.convolution2d(net_aux, 128, [3, 3], stride=1, scope='pfld_conv2')
        print(net_aux.name, net_aux.get_shape())
        net_aux = slim.convolution2d(net_aux, 32, [3, 3], stride=2, scope='pfld_conv3')
        print(net_aux.name, net_aux.get_shape())
        net_aux = slim.convolution2d(net_aux, 128, [7, 7], stride=1, scope='pfld_conv4')
        print(net_aux.name, net_aux.get_shape())
        net_aux = slim.max_pool2d(net_aux, kernel_size=[3, 3], stride=1, scope='pool1', padding='SAME')
        print(net_aux.name, net_aux.get_shape())
        net_aux = slim.flatten(net_aux)
        print(net_aux.name, net_aux.get_shape())
        fc1 = slim.fully_connected(net_aux, num_outputs=32, activation_fn=None, scope='pfld_fc1')
        print(fc1.name, fc1.get_shape())
        euler_angles_pre = slim.fully_connected(fc1, num_outputs=3, activation_fn=None, scope='pfld_fc2')
        print(euler_angles_pre.name, euler_angles_pre.get_shape())
        # pfld_fc2/BatchNorm/Reshape_1:0

        return euler_angles_pre


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
    features, landmarks_pre = pfld_backbone(input, args.weight_decay, batch_norm_params, args.num_labels, args.depth_multi)
    # loss
    landmarks_loss = tf.reduce_sum(tf.square(landmarks_pre - landmark), axis=1)
    landmarks_loss = tf.reduce_mean(landmarks_loss)

    euler_angles_pre = pfld_auxiliary(features, args.weight_decay, batch_norm_params)

    print("==========finish define graph===========")

    # return landmarks_loss, landmarks, heatmap_loss, HeatMaps
    return landmarks_pre, landmarks_loss, euler_angles_pre


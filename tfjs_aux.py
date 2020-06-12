# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import LandmarkImage, LandmarkImage_98

import time


def conv2d(net, stride, num_channel, kernel, scope):
    net = slim.conv2d(net, num_channel, [kernel, kernel], stride=stride, scope=scope)
    print(net.name, net.get_shape())

    return net


def depthwiseSeparableConv(net, _s_size, num_channel, scope):
    net = slim.separable_convolution2d(net,
                                    num_outputs=None,
                                    stride=_s_size,
                                    depth_multiplier=1,
                                    kernel_size=[3, 3],
                                    scope=scope+'/depthwise_conv')
    net = slim.convolution2d(net,
                            num_channel, # num_output
                            kernel_size=[1, 1],
                            scope=scope+'/pointwise_conv')

    return net


def denseBlock(input, denseblockparams, isfirstlayer=False):
    isScaleDown = True
    if isfirstlayer:
        _s_size = 2 if isScaleDown else 1
        out1 = conv2d(input, _s_size, num_channel=denseblockparams["out"], kernel=3, scope='conv1')
        out1 = tf.nn.relu(out1)
    else:
        _s_size = 2 if isScaleDown else 1
        out1 = depthwiseSeparableConv(input, _s_size, num_channel=denseblockparams["out"], scope=denseblockparams["name"]+"out1")
    print(out1.name, out1.get_shape())

    _s_size = 1
    out2 = depthwiseSeparableConv(out1, _s_size, num_channel=denseblockparams["out"], scope=denseblockparams["name"]+"out2")
    print(out2.name, out2.get_shape())

    in3 = tf.nn.relu(tf.add(out1, out2))
    out3 = depthwiseSeparableConv(in3, _s_size, num_channel=denseblockparams["out"], scope=denseblockparams["name"]+"out3")
    print(out3.name, out3.get_shape())

    in4 = tf.nn.relu(tf.add(out1, tf.add(out2, out3)))
    out4 = depthwiseSeparableConv(in4, _s_size, num_channel=denseblockparams["out"], scope=denseblockparams["name"]+"out4")
    out = tf.nn.relu(tf.add(out1, tf.add(out2, tf.add(out3, out4))))
    print(out.name, out.get_shape())

    return out


def tfjs_inference(input, dense_params, batch_norm_params, weight_decay, num_labels):
    features = {}
    with slim.arg_scope(
        [slim.conv2d, slim.separable_convolution2d, slim.convolution2d],
        # activation_fn=tf.nn.relu6,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
        padding='SAME'
        ):
        print('PFLD input shape({}): {}'.format(input.name, input.get_shape()))
        dense0 = denseBlock(input, dense_params["dense0"], True)
        print(dense0.name, dense0.get_shape())
        dense1 = denseBlock(dense0, dense_params["dense1"])
        print(dense1.name, dense1.get_shape())
        features['auxiliary_input'] = dense1
        dense2 = denseBlock(dense1, dense_params["dense2"])
        print(dense2.name, dense2.get_shape())
        dense3 = denseBlock(dense2, dense_params["dense3"])
        print(dense3.name, dense3.get_shape())
        pooled = slim.avg_pool2d(dense3, 7, 2, 'VALID')
        #pooled = slim.avg_pool2d(dense6, 2, 2, 'VALID')
        print(pooled.name, pooled.get_shape())
        flattened = slim.flatten(pooled)
        landmarks = slim.fully_connected(flattened, num_outputs=num_labels*2, activation_fn=None, scope='fc')
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
    dense_params = {
        "dense0": {"in": 3, "out": 32, "name": 'dense0'}, # inc, outc, isfirst, name
        "dense1": {"in": 32, "out": 64, "name": 'dense1'},
        "dense2": {"in": 64, "out": 128, "name": 'dense2'},
        "dense3": {"in": 128, "out": 256, "name": 'dense3'},
    }

    landmark_dim = int(landmark.get_shape()[-1])
    print("labels; ", args.num_labels)
    time.sleep(3)
    features, landmarks_pre = tfjs_inference(input, dense_params, batch_norm_params, args.weight_decay, args.num_labels)
    # loss
    landmarks_loss = tf.reduce_sum(tf.square(landmarks_pre - landmark), axis=1)
    landmarks_loss = tf.reduce_mean(landmarks_loss)

    euler_angles_pre = pfld_auxiliary(features, args.weight_decay, batch_norm_params)

    print("==========finish define graph===========")

    return landmarks_pre, landmarks_loss, euler_angles_pre

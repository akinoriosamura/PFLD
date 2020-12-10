# -*- coding: UTF-8 -*-
import numpy as np
import math
import random
import time
import gc
import sys
import os
import tensorflow as tf
import tensorflow_addons as tfa
from generate_data_tfrecords_tf2 import TfrecordsLoader
from tensorflow.keras.layers import Add, Flatten, Concatenate, Dense, Conv2D, DepthwiseConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras import Model
tf.keras.backend.set_floatx('float32')


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, is_first, s, channel, in_channel, up_sample, name='bblock'):
        super(Bottleneck, self).__init__()
        self.is_first = is_first
        self.s = s
        self.channel = channel
        self.in_channel = in_channel
        self.tchannel = in_channel * up_sample
        self.vname = name
        self.conv2d_1 = self.conv2d(self.tchannel, 1, 1, padding='same', name=self.vname+'_conv2d_1')
        self.bn_1 = BatchNormalization(name=self.vname+'_bn1')
        self.dwconv2d_2 = DepthwiseConv2D(3, strides=(self.s, self.s), depth_multiplier=1, padding='same', name=self.vname+'_dwconv2d_1')
        self.bn_2 = BatchNormalization(name=self.vname+'_bn2')
        self.conv2d_3 = self.conv2d(self.channel, 1, 1, padding='same', name=self.vname+'_conv2d_3')
        self.bn_3 = BatchNormalization(name=self.vname+'_bn3')


    def conv2d(self, filters, k, s, padding='same', name='none'):
        return Conv2D(
            filters,
            k,
            strides=(s, s),
            padding=padding,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            trainable=True,
            name=name
        )

    def call(self, inputs, training=None):
        x = self.conv2d_1(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dwconv2d_2(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2d_3(x)
        x = self.bn_3(x, training=training)
        if (self.s == 1) and (self.channel == self.in_channel):
            x = Add()([x, inputs])

        return x


class PFLDBackbone(Model):
    def __init__(self, num_labels, img_size, depth_multi):
        super(PFLDBackbone, self).__init__()
        self.num_labels = num_labels
        self.img_size = img_size
        self.depth_multi = depth_multi
        # 112*112*3 / conv3*3 / c:64,n:1,s:2
        self.conv2d_1 = self.conv2d(64, 3, 2, name='conv1')
        self.bn_1 = BatchNormalization(name='bn1')
        self.relu1 = tf.nn.relu
        # 56*56*64 / depthwiseconv3*3 / c:64,n:1,s:1
        dw_channel = self.depth(64)
        self.dw_conv2d_2 = self.dwconv2d(3, 1, name='dwise2')
        self.bn_2 = BatchNormalization(name='bn2')
        self.relu2 = tf.nn.relu
        # 56*56*64 / InverseBottleneck / up_s:2,c:64,n:5,s:2
        bb1_channel = self.depth(64)
        self.bottleneck_1_1 = Bottleneck(is_first=True, s=2, channel=bb1_channel, in_channel=dw_channel, up_sample=2, name='bblock1_1')
        self.bottleneck_1_2 = Bottleneck(is_first=False, s=1, channel=bb1_channel, in_channel=dw_channel, up_sample=2, name='bblock1_2')
        self.bottleneck_1_3 = Bottleneck(is_first=False, s=1, channel=bb1_channel, in_channel=dw_channel, up_sample=2, name='bblock1_3')
        self.bottleneck_1_4 = Bottleneck(is_first=False, s=1, channel=bb1_channel, in_channel=dw_channel, up_sample=2, name='bblock1_4')
        self.bottleneck_1_5 = Bottleneck(is_first=False, s=1, channel=bb1_channel, in_channel=dw_channel, up_sample=2, name='bblock1_5')
        # 28*28*64 / InverseBottleneck / up_s:2,c:128,n:1,s:2
        bb2_channel = self.depth(128)
        self.bottleneck_2_1 = Bottleneck(is_first=True, s=2, channel=bb2_channel, in_channel=bb1_channel, up_sample=2, name='bblock2_1')
        # 14*14*128 / InverseBottleneck / up_s:4,c:128,n:6,s:1
        bb3_channel = self.depth(128)
        self.bottleneck_3_1 = Bottleneck(is_first=True, s=1, channel=bb3_channel, in_channel=bb2_channel, up_sample=4, name='bblock3_1')
        self.bottleneck_3_2 = Bottleneck(is_first=False, s=1, channel=bb3_channel, in_channel=bb2_channel, up_sample=4, name='bblock3_2')
        self.bottleneck_3_3 = Bottleneck(is_first=False, s=1, channel=bb3_channel, in_channel=bb2_channel, up_sample=4, name='bblock3_3')
        self.bottleneck_3_4 = Bottleneck(is_first=False, s=1, channel=bb3_channel, in_channel=bb2_channel, up_sample=4, name='bblock3_4')
        self.bottleneck_3_5 = Bottleneck(is_first=False, s=1, channel=bb3_channel, in_channel=bb2_channel, up_sample=4, name='bblock3_5')
        self.bottleneck_3_6 = Bottleneck(is_first=False, s=1, channel=bb3_channel, in_channel=bb2_channel, up_sample=4, name='bblock3_6')
        # 14*14*128 / InverseBottleneck / up_s:2,c:16,n:1,s:1
        bb4_channel = self.depth(16)
        self.bottleneck_4_1 = Bottleneck(is_first=True, s=1, channel=bb4_channel, in_channel=bb3_channel, up_sample=2, name='bblock4_1')
        # 14*14*16 / conv3*3 / c:32,n:1,s:2
        c2_channel = self.depth(32)
        self.conv2d_3 = self.conv2d(c2_channel, 3, 2, name='conv3')
        self.bn_3 = BatchNormalization(name='bn3')
        self.relu3 = tf.nn.relu
        # 7*7*32 / conv7*7 / c:128,n:1,s:1
        c3_channel = self.depth(128)
        self.conv2d_4 = self.conv2d(c3_channel, 7, 1, padding='valid', name='conv4')
        self.bn_4 = BatchNormalization(name='bn4')
        self.relu4 = tf.nn.relu

        p1_size = self.img_size / 8
        self.avg_pool1 = self.avg_pool2d(p1_size, s=1, name='avgpool1')
        p2_size = self.img_size / 16
        self.avg_pool2 = self.avg_pool2d(p2_size, s=1, name='avgpool2')
        self.flatten = Flatten()
        # 1*1*128
        self.concat = Concatenate(axis=1)
        self.fc = self.dense(num_labels*2, name='fc')


    def depth(self, d, min_depth=8):
        return max(int(d * self.depth_multi), min_depth)

    def conv2d(self, filters, k, s, padding='same', name='none'):
        return Conv2D(
            filters,
            k,
            strides=(s, s),
            padding=padding,
            activation=None,
            # activation=tf.nn.relu6,
            # activation=tf.nn.relu,
            # activation=mish,
            use_bias=True,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            trainable=True,
            name=name
        )

    def dwconv2d(self, k, s, name='none'):
        return DepthwiseConv2D(
            k,
            strides=(s, s),
            activation=None,
            # activation=tf.nn.relu6,
            # activation=tf.nn.relu,
            # activation=mish,
            use_bias=True,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            trainable=True,
            name=name
        )

    def avg_pool2d(self, ps, s, trainable=True, name='none'):
        return AveragePooling2D(
            pool_size=(ps, ps),
            strides=(s, s),
            trainable=trainable,
            name=name
        )

    def dense(self, units, trainable=True, name='none'):
        return Dense(
            units,
            use_bias=True,
            activation=None, # tf.nn.relu6,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            trainable=trainable,
            name=name
        )


    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


    def call(self, input, training=True):
        print("=== start network ===")
        # import pdb;pdb.set_trace()
        print('PFLD input shape({}): {}'.format(input.name, input.shape))
        # 112*112*3 / conv3*3 / c:64,n:1,s:2
        x = self.conv2d_1(input)
        x = self.bn_1(x, training=training)
        x = self.relu1(x)
        # import pdb;pdb.set_trace()
        print(x.name, x.get_shape())
        # 56*56*64 / depthwiseconv3*3 / c:64,n:1,s:1
        x = self.dw_conv2d_2(x)
        x = self.bn_2(x, training=training)
        x = self.relu2(x)
        print(x.name, x.get_shape())
        # 56*56*64 / InverseBottleneck / up_s:2,c:64,n:5,s:2
        x = self.bottleneck_1_1(x, training=training)
        x = self.bottleneck_1_2(x, training=training)
        x = self.bottleneck_1_3(x, training=training)
        x = self.bottleneck_1_4(x, training=training)
        x = self.bottleneck_1_5(x, training=training)
        features3 = x
        print("layer features3 is ") # 112:27,27,64, 84: 20,20,64
        print(x.name, x.get_shape())
        # 28*28*64 / InverseBottleneck / up_s:2,c:128,n:1,s:2
        x = self.bottleneck_2_1(x, training=training)
        print(x.name, x.get_shape())
        # 14*14*128 / InverseBottleneck / up_s:4,c:128,n:6,s:1
        x = self.bottleneck_3_1(x, training=training)
        x = self.bottleneck_3_2(x, training=training)
        x = self.bottleneck_3_3(x, training=training)
        x = self.bottleneck_3_4(x, training=training)
        x = self.bottleneck_3_5(x, training=training)
        x = self.bottleneck_3_6(x, training=training)
        print(x.name, x.get_shape())
        # 14*14*128 / InverseBottleneck / up_s:2,c:16,n:1,s:1
        x = self.bottleneck_4_1(x, training=training)
        features6 = x
        print("layer features6 is ")
        print(x.name, x.get_shape())
        # 14*14*16 / conv3*3 / c:32,n:1,s:2
        x = self.conv2d_3(x)
        x = self.bn_3(x, training=training)
        x = self.relu3(x)
        features7 = x
        print("layer features7 is ")
        print(x.name, x.get_shape())
        # 7*7*32 / conv7*7 / c:128,n:1,s:1
        x = self.conv2d_4(x)
        x = self.bn_4(x, training=training)
        x = self.relu4(x)
        features8 = x
        print("layer features8 is ")
        print(x.name, x.get_shape())
        # get avg pool
        avg_pool1 = self.avg_pool1(features6)
        print(avg_pool1.name, avg_pool1.get_shape())
        avg_pool2 = self.avg_pool2(features7)
        print(avg_pool2.name, avg_pool2.get_shape())
        s1 = self.flatten(avg_pool1)
        s2 = self.flatten(avg_pool2)
        # 1*1*128
        s3 = self.flatten(features8)
        multi_scale = self.concat([s1, s2, s3])
        print(multi_scale.name, multi_scale.get_shape())
        landmarks = self.fc(multi_scale)
        print("last layer name")
        print(landmarks.name, landmarks.get_shape())

        return features3, landmarks


class PFLDAuxiliary(Model):
    def __init__(self, num_labels, img_size, depth_multi):
        super(PFLDAuxiliary, self).__init__()
        self.num_labels = num_labels
        self.img_size = img_size
        self.depth_multi = depth_multi
        self.conv2d_1 = self.conv2d(128, 3, 2, name='conv1')
        self.bn_1 = BatchNormalization(name='bn1')
        self.relu1 = tf.nn.relu
        self.conv2d_2 = self.conv2d(128, 3, 1, name='conv2')
        self.bn_2 = BatchNormalization(name='bn2')
        self.relu2 = tf.nn.relu
        self.conv2d_3 = self.conv2d(32, 3, 2, name='conv3')
        self.bn_3 = BatchNormalization(name='bn3')
        self.relu3 = tf.nn.relu
        self.conv2d_4 = self.conv2d(128, 7, 1, name='conv4')
        self.bn_4 = BatchNormalization(name='bn4')
        self.relu4 = tf.nn.relu
        self.maxpool2d = self.maxpool2d(3, s=1, name='maxpool1')
        self.flatten = Flatten()
        self.fc1 = self.dense(32, name='fc1')
        self.fc2 = self.dense(3, name='fc2')

    def conv2d(self, filters, k, s, padding='same', name='none'):
        return Conv2D(
            filters,
            k,
            strides=(s, s),
            padding=padding,
            activation=None,
            # activation=tf.nn.relu6,
            # activation=tf.nn.relu,
            # activation=mish,
            use_bias=True,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            trainable=True,
            name=name
        )

    def dense(self, units, trainable=True, name='none'):
        return Dense(
            units,
            use_bias=True,
            activation=None, # tf.nn.relu6,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            trainable=trainable,
            name=name
        )

    def maxpool2d(self, ps, s, padding='same', trainable=True, name='none'):
        return MaxPooling2D(
            pool_size=(ps, ps),
            strides=(s, s),
            padding=padding,
            trainable=trainable,
            name=name
        )

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


    def call(self, input, training=True):
        print("=== start aux network ===")
        # import pdb;pdb.set_trace()
        print('PFLD aux input shape({}): {}'.format(input.name, input.shape))
        x = self.conv2d_1(input)
        x = self.bn_1(x, training=training)
        x = self.relu1(x)
        print(x.name, x.get_shape())
        x = self.conv2d_2(x)
        x = self.bn_2(x, training=training)
        x = self.relu2(x)
        print(x.name, x.get_shape())
        x = self.conv2d_3(x)
        x = self.bn_3(x, training=training)
        x = self.relu3(x)
        print(x.name, x.get_shape())
        x = self.conv2d_4(x)
        x = self.bn_4(x, training=training)
        x = self.relu4(x)
        print(x.name, x.get_shape())
        x = self.maxpool2d(x)
        print(x.name, x.get_shape())
        x = self.flatten(x)
        print(x.name, x.get_shape())
        fc1 = self.fc1(x)
        print(fc1.name, fc1.get_shape())
        euler_angles_pre = self.fc2(fc1)
        print(euler_angles_pre.name, euler_angles_pre.get_shape())
        # pfld_fc2/BatchNorm/Reshape_1:0

        return euler_angles_pre

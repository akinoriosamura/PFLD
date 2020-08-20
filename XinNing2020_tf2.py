# -*- coding: UTF-8 -*-
import numpy as np
import math
import random
import time
import gc
import sys
import os
import tensorflow as tf
from generate_data_tfrecords_tf2 import TfrecordsLoader
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras import Model
tf.keras.backend.set_floatx('float32')


class XinNingNetwork(Model):
    def __init__(self, num_labels, img_size, meat_shape, train_phase='stage1'):
        super(XinNingNetwork, self).__init__()
        self.num_labels = num_labels
        self.img_size = img_size
        self.meat_shape = tf.cast(meat_shape, dtype=tf.float32)
        if train_phase=='stage1':
            self.train_stage1 = True
        elif train_phase=='stage2':
            self.train_stage1 = False
        else:
            print("error model initializer train phase")
            exit()
        ###### stage1 ######
        # 112*112*3(1) / conv3*3 / c:16,n:1,s:2
        self.conv1_1 = self.conv2d(16, 3, 2, trainable=self.train_stage1, name='conv1_1')
        self.bn1_1 = BatchNormalization(trainable=self.train_stage1, name='bn1_1')
        # 56*56*16 / conv3*3 / c:32,n:1,s:2
        self.conv1_2 = self.conv2d(32, 3, 2, trainable=self.train_stage1, name='conv1_2')
        self.bn1_2 = BatchNormalization(trainable=self.train_stage1, name='bn1_2')
        # 28*28*32 / pool2*2 / c:32,n:1,s:2
        self.pool1_2 = self.maxpool2d(2, 2, trainable=self.train_stage1, name='pool1_2')
        # 14*14*32 / conv3*3 / c:64,n:1,s:2
        self.conv1_2_1 = self.conv2d(64, 3, 2, trainable=self.train_stage1, name='conv1_2_1')
        self.bn1_2_1 = BatchNormalization(trainable=self.train_stage1, name='bn1_2_1')
        # 7*7*64 / global_pool / c:64,n:1
        self.pool1_2_1 = GlobalAveragePooling2D(trainable=self.train_stage1, name='pool1_2_1')
        # 14*14*32 / conv3*3 / c:64,n:1,s:2
        self.conv1_3 = self.conv2d(64, 3, 2, trainable=self.train_stage1, name='conv1_3')
        self.bn1_3 = BatchNormalization(trainable=self.train_stage1, name='bn1_3')
        # 7*7*64 / pool2*2 / c:64,n:1,s:2
        self.pool1_3 = self.maxpool2d(2, 2, trainable=self.train_stage1, name='pool1_3')
        # 4*4*64 / conv3*3 / c:64,n:1,s:2
        self.conv1_3_1 = self.conv2d(64, 3, 2, trainable=self.train_stage1, name='conv1_3_1')
        self.bn1_3_1 = BatchNormalization(trainable=self.train_stage1, name='bn1_3_1')
        # 2*2*64 / global_pool / c:64,n:1
        self.pool1_3_1 = GlobalAveragePooling2D(trainable=self.train_stage1, name='pool1_3_1')
        # 4*4*64 / conv3*3 / c:64,n:1,s:2
        self.conv1_4 = self.conv2d(64, 3, 2, trainable=self.train_stage1, name='conv1_4')
        self.bn1_4 = BatchNormalization(trainable=self.train_stage1, name='bn1_4')
        # 2*2*64 / global_pool / c:64,n:1
        self.pool1_4_1 = GlobalAveragePooling2D(trainable=self.train_stage1, name='pool1_4_1')
        # 1*1*64*3() / concat / 1*1*192
        # 1*1*192 / fc / 1*136
        self.fc1 = self.dense(num_labels*2, trainable=self.train_stage1, name='fc1')

        ###### stage2 ######
        # 112*112*1*2 / concat / 112*112*2
        # 112*112*2 / conv3*3 / c:8,n:1,s:2
        self.conv2_1 = self.conv2d(8, 3, 2, name='conv2_1')
        self.bn2_1 = BatchNormalization(name='bn2_1')
        # 56*56*8 / pool3*3 / c:28,n:1,s:2
        self.pool2_1 = self.maxpool2d(3, 2, name='pool2_1')
        # 28*28*8 / conv3*3 / c:16,n:1,s:1
        self.conv2_2 = self.conv2d(16, 3, 1, name='conv2_2')
        self.bn2_2 = BatchNormalization(name='bn2_2')
        # 28*28*16 / pool3*3 / c:16,n:1,s:2
        self.pool2_2 = self.maxpool2d(3, 2, name='pool2_2')
        # 14*14*16 / conv3*3 / c:64,n:1,s:1
        self.conv2_2_1 = self.conv2d(64, 3, 1, name='conv2_2_1')
        self.bn2_2_1 = BatchNormalization(name='bn2_2_1')
        # 14*14*64 / global_pool / c:64,n:1
        self.pool2_2_1 = GlobalAveragePooling2D(name='pool2_2_1')
        # 14*14*16 / conv3*3 / c:64,n:1,s:2
        self.conv2_3 = self.conv2d(64, 3, 2, name='conv2_3')
        self.bn2_3 = BatchNormalization(name='bn2_3')
        # 7*7*64 / pool3*3 / c:64,n:1,s:2
        self.pool2_3 = self.maxpool2d(3, 2, padding='valid', name='pool2_3')
        # 3*3*64 / conv3*3 / c:64,n:1,s:1
        self.conv2_3_1 = self.conv2d(64, 3, 1, name='conv2_3_1')
        self.bn2_3_1 = BatchNormalization(name='bn2_3_1')
        # 3*3*64 / global_pool / c:64,n:1
        self.pool2_3_1 = GlobalAveragePooling2D(name='pool2_3_1')
        # 3*3*64 / conv3*3 / c:64,n:1,s:2
        self.conv2_4 = self.conv2d(64, 3, 2, name='conv2_4')
        self.bn2_4 = BatchNormalization(name='bn2_4')
        # 2*2*64 / global_pool / c:64,n:1
        self.pool2_4_1 = GlobalAveragePooling2D(name='pool2_4_1')
        # 1*1*64*3 / concat / 1*1*192
        # 1*1*192 / fc / 1*136
        self.fc2 = self.dense(num_labels*2, name='fc2')

    def conv2d(self, filters, k, s, padding='same', use_bias=False, trainable=True, name='none'):
        return Conv2D(
            filters,
            k,
            strides=(s, s),
            padding=padding,
            activation=tf.nn.relu6,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
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

    def dense(self, units, trainable=True, name='none'):
        return Dense(
            units,
            use_bias=True,
            activation=tf.nn.relu6,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            trainable=trainable,
            name=name
        )

    """
    # by numpy
    def label_heatmap(self, land):
        # land: [2]
        x_range = np.range(0, 112)
        y_range = np.range(0, 112)
        xx, yy = np.meshgrid(x_range, y_range)
        land_x = land[0] * 112
        land_y = land[1] * 112
        _xx = (xx.astype(tf.float32) - land_x) ** 2
        _yy = (yy.astype(tf.float32) - land_y) ** 2
        d2 = _xx + _yy
        heatmap_tmp = np.exp(-d2)

        return heatmap_tmp

    def img_heatmap(self, land):
        # land: [68, 2]
        one_label_heatmap = map(self.label_heatmap, land)
        labels_heatmap = np.sum(one_label_heatmap, axis=0)

        return labels_heatmap

    def HeatMap(self, output):
        # ?*136 / transform / ?*112*112*1
        # import pdb;pdb.set_trace()
        # landmark: [?, 68, 2]
        output = output.numpy()
        landmark = output.reshape(-1, self.num_labels, 2)
        print(landmark.name, landmark.shape)
        heatmap = map(self.img_heatmap, landmark)
        heatmap = np.expand_dims(heatmap, -1)
        print(heatmap.name, heatmap.shape)

        return heatmap, [landmark, heatmap]
    """

    # by tf
    def label_heatmap(self, label_input):
        # land: [2]
        land = label_input[0]
        ms = label_input[1]
        x_range = tf.range(0, self.img_size)
        y_range = tf.range(0, self.img_size)
        xx, yy = tf.meshgrid(x_range, y_range)
        # land is normalized so return by img size multiply
        land_x = tf.math.multiply(tf.add(land[0], ms[0]), self.img_size)
        land_y = tf.math.multiply(tf.add(land[1], ms[1]), self.img_size)
        _xx = tf.pow(tf.subtract(tf.cast(xx, dtype=tf.float32), land_x), 2)
        _yy = tf.pow(tf.subtract(tf.cast(yy, dtype=tf.float32), land_y), 2)
        d2 = tf.add(_xx, _yy)
        heatmap_tmp = tf.math.exp(
            tf.math.negative(d2), name="one_label_heatmap")

        return heatmap_tmp

    def img_heatmap(self, land):
        # land: [68, 2]
        _meat_shapes = tf.reshape(self.meat_shape, [68, 2])
        one_label_heatmap = tf.vectorized_map(self.label_heatmap, (land, _meat_shapes))
        labels_heatmap = tf.reduce_sum(
            one_label_heatmap, 0, name="label_heatmap")
        max_v = tf.reduce_max(labels_heatmap)
        labels_heatmap = tf.math.divide(labels_heatmap, max_v)
        return labels_heatmap

    def HeatMap(self, output):
        # ?*136 / transform / ?*112*112*1
        # import pdb;pdb.set_trace()
        # landmark: [?, 68, 2]
        landmark = tf.reshape(output, [-1, 68, 2])
        heatmap = tf.vectorized_map(self.img_heatmap, landmark)
        heatmap = tf.expand_dims(heatmap, -1, name="heatmap")

        return heatmap, [landmark, heatmap]

    def call(self, input, training=True):
        print("=== start network ===")
        # import pdb;pdb.set_trace()
        print('PFLD input shape({}): {}'.format(input.name, input.shape))
        ###### stage1 ######
        print("=== start stage 1 ===")
        # 112*112*3(1) / conv3*3 / c:16,n:1,s:2
        _conv1_1 = self.bn1_1(self.conv1_1(input), training=training)
        print(_conv1_1.name, _conv1_1.shape)
        # 56*56*16 / conv3*3 / c:32,n:1,s:2
        _conv1_2 = self.bn1_2(self.conv1_2(_conv1_1), training=training)
        print(_conv1_2.name, _conv1_2.shape)
        # 28*28*32 / pool2*2 / c:32,n:1,s:2
        _pool1_2 = self.pool1_2(_conv1_2)
        print(_pool1_2.name, _pool1_2.shape)
        # 14*14*32 / conv3*3 / c:64,n:1,s:2
        _conv1_2_1 = self.bn1_2_1(self.conv1_2_1(_pool1_2), training=training)
        print(_conv1_2_1.name, _conv1_2_1.shape)
        # 7*7*64 / global_pool / c:64,n:1
        _pool1_2_1 = self.pool1_2_1(_conv1_2_1)
        print(_pool1_2_1.name, _pool1_2_1.shape)
        # 14*14*32 / conv3*3 / c:64,n:1,s:2
        _conv1_3 = self.bn1_3(self.conv1_3(_pool1_2), training=training)
        print(_conv1_3.name, _conv1_3.shape)
        # 7*7*64 / pool2*2 / c:64,n:1,s:2
        _pool1_3 = self.pool1_3(_conv1_3)
        print(_pool1_3.name, _pool1_3.shape)
        # 4*4*64 / conv3*3 / c:64,n:1,s:2
        _conv1_3_1 = self.bn1_3_1(self.conv1_3_1(_pool1_3), training=training)
        print(_conv1_3_1.name, _conv1_3_1.shape)
        # 2*2*64 / global_pool / c:64,n:1
        _pool1_3_1 = self.pool1_3_1(_conv1_3_1)
        print(_pool1_3_1.name, _pool1_3_1.shape)
        # 4*4*64 / conv3*3 / c:64,n:1,s:2
        _conv1_4 = self.bn1_4(self.conv1_4(_pool1_3), training=training)
        print(_conv1_4.name, _conv1_4.shape)
        # 2*2*64 / global_pool / c:64,n:1
        _pool1_4_1 = self.pool1_4_1(_conv1_4)
        print(_pool1_4_1.name, _pool1_4_1.shape)
        # 1*1*64*3() / concat / 1*1*192
        _concatted_1 = tf.concat([_pool1_2_1, _pool1_3_1, _pool1_4_1], 1)
        print(_concatted_1.name, _concatted_1.shape)
        # 1*192 / fc / 1*136
        _output_1 = self.fc1(_concatted_1)
        print(_output_1.name, _output_1.shape)
        # 1*136 / transform / 112*112*1
        _heatmap, _heat_values = self.HeatMap(_output_1)
        print(_heatmap.name, _heatmap.shape)
        print("=== finish stage 1 ===")
        if self.train_stage1:
            # if True:
            print("=== finish train in stage 1 ===")
            return [_output_1], _heat_values

        ###### stage2 ######
        print("=== start stage 2 ===")
        # import pdb;pdb.set_trace()
        # 112*112*1*2 / concat / 112*112*2
        _concatted_2 = tf.concat([input, _heatmap], 3)
        print(_concatted_2.name, _concatted_2.shape)
        # 112*112*2 / conv3*3 / c:8,n:1,s:2
        _conv2_1 = self.bn2_1(self.conv2_1(_concatted_2), training=training)
        print(_conv2_1.name, _conv2_1.shape)
        # 56*56*8 / pool3*3 / c:28,n:1,s:2
        _pool2_1 = self.pool2_1(_conv2_1)
        print(_pool2_1.name, _pool2_1.shape)
        # 28*28*8 / conv3*3 / c:16,n:1,s:1
        _conv2_2 = self.bn2_2(self.conv2_2(_pool2_1), training=training)
        print(_conv2_2.name, _conv2_2.shape)
        # 28*28*16 / pool3*3 / c:16,n:1,s:2
        _pool2_2 = self.pool2_2(_conv2_2)
        print(_pool2_2.name, _pool2_2.shape)
        # 14*14*16 / conv3*3 / c:64,n:1,s:1
        _conv2_2_1 = self.bn2_2_1(self.conv2_2_1(_pool2_2), training=training)
        print(_conv2_2_1.name, _conv2_2_1.shape)
        # 14*14*64 / global_pool / c:64,n:1
        _pool2_2_1 = self.pool2_2_1(_conv2_2_1)
        print(_pool2_2_1.name, _pool2_2_1.shape)
        # 14*14*16 / conv3*3 / c:64,n:1,s:2
        _conv2_3 = self.bn2_3(self.conv2_3(_pool2_2), training=training)
        print(_conv2_3.name, _conv2_3.shape)
        # 7*7*64 / pool3*3 / c:64,n:1,s:2
        _pool2_3 = self.pool2_3(_conv2_3)
        print(_pool2_3.name, _pool2_3.shape)
        # 3*3*64 / conv3*3 / c:64,n:1,s:1
        _conv2_3_1 = self.bn2_3_1(self.conv2_3_1(_pool2_3), training=training)
        print(_conv2_3_1.name, _conv2_3_1.shape)
        # 3*3*64 / global_pool / c:64,n:1
        _pool2_3_1 = self.pool2_3_1(_conv2_3_1)
        print(_pool2_3_1.name, _pool2_3_1.shape)
        # 3*3*64 / conv3*3 / c:64,n:1,s:2
        _conv2_4 = self.bn2_4(self.conv2_4(_pool2_3), training=training)
        print(_conv2_4.name, _conv2_4.shape)
        # 2*2*64 / global_pool / c:64,n:1
        _pool2_4_1 = self.pool2_4_1(_conv2_4)
        print(_pool2_4_1.name, _pool2_4_1.shape)
        # 1*1*64*3 / concat / 1*1*192
        _concatted_3 = tf.concat([_pool2_2_1, _pool2_3_1, _pool2_4_1], 1)
        print(_concatted_3.name, _concatted_3.shape)
        # 1*1*192 / fc / 1*136
        _output_2 = self.fc2(_concatted_3)
        print("last layer name")
        print(_output_2.name, _output_2.shape)
        print("=== finish stage 2 ===")
        print("=== finish train in stage 2 ===")

        return [_output_1, _output_2], _heat_values

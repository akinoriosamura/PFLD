# -*- coding: UTF-8 -*-
import tensorflow as tf
from generate_data_tfrecords_tf2 import TfrecordsLoader
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras import Model
tf.keras.backend.set_floatx('float32')

import os
import sys
import gc
import time
import random
import math
import numpy as np


class XinNingNetwork(Model):
    def __init__(self, num_labels, img_size):
        super(XinNingNetwork, self).__init__()
        self.num_labels = num_labels
        self.img_size = img_size
        ###### stage1 ######
        # 112*112*3(1) / conv3*3 / c:16,n:1,s:2
        self.conv1_1 = self.conv2d(16, 3, 2)
        # 56*56*16 / conv3*3 / c:32,n:1,s:2
        self.conv1_2 = self.conv2d(32, 3, 2)
        # 28*28*32 / pool2*2 / c:32,n:1,s:2
        self.pool1_2 = self.maxpool2d(2, 2)
        # 14*14*32 / conv3*3 / c:64,n:1,s:2
        self.conv1_2_1 = self.conv2d(64, 3, 2)
        # 7*7*64 / global_pool / c:64,n:1
        self.pool1_2_1 = GlobalAveragePooling2D()
        # 14*14*32 / conv3*3 / c:64,n:1,s:2
        self.conv1_3 = self.conv2d(64, 3, 2)
        # 7*7*64 / pool2*2 / c:64,n:1,s:2
        self.pool1_3 = self.maxpool2d(2, 2)
        # 4*4*64 / conv3*3 / c:64,n:1,s:2
        self.conv1_3_1 = self.conv2d(64, 3, 2)
        # 2*2*64 / global_pool / c:64,n:1
        self.pool1_3_1 = GlobalAveragePooling2D()
        # 4*4*64 / conv3*3 / c:64,n:1,s:2
        self.conv1_4 = self.conv2d(64, 3, 2)
        # 2*2*64 / global_pool / c:64,n:1
        self.pool1_4_1 = GlobalAveragePooling2D()
        # 1*1*64*3() / concat / 1*1*192
        # flatten
        self.flatten1 = Flatten()
        # 1*1*192 / fc / 1*136
        self.fc1 = self.dense(num_labels*2)

        ###### stage2 ######
        # 112*112*1*2 / concat / 112*112*2
        # 112*112*2 / conv3*3 / c:8,n:1,s:2
        self.conv2_1 = self.conv2d(8, 3, 2)
        # 56*56*8 / pool3*3 / c:28,n:1,s:2
        self.pool2_1 = self.maxpool2d(3, 2)
        # 28*28*8 / conv3*3 / c:16,n:1,s:1
        self.conv2_2 = self.conv2d(16, 3, 1)
        # 28*28*16 / pool3*3 / c:16,n:1,s:2
        self.pool2_2 = self.maxpool2d(3, 2)
        # 14*14*16 / conv3*3 / c:64,n:1,s:1
        self.conv2_2_1 = self.conv2d(64, 3, 1)
        # 14*14*64 / global_pool / c:64,n:1
        self.pool2_2_1 = GlobalAveragePooling2D()
        # 14*14*16 / conv3*3 / c:64,n:1,s:2
        self.conv2_3 = self.conv2d(64, 3, 2)
        # 7*7*64 / pool3*3 / c:64,n:1,s:2
        self.pool2_3 = self.maxpool2d(3, 2)
        # 3*3*64 / conv3*3 / c:64,n:1,s:1
        self.conv2_3_1 = self.conv2d(64, 3, 1)
        # 3*3*64 / global_pool / c:64,n:1
        self.pool2_3_1 = GlobalAveragePooling2D()
        # 3*3*64 / conv3*3 / c:64,n:1,s:2
        self.conv2_4 = self.conv2d(64, 3, 2)
        # 2*2*64 / global_pool / c:64,n:1
        self.pool2_4_1 = GlobalAveragePooling2D()
        # 1*1*64*3 / concat / 1*1*192
        # flatten / 1*192
        self.flatten2 = Flatten()
        # 1*1*192 / fc / 1*136
        self.fc2 = self.dense(num_labels*2)

    def conv2d(self, filters, k, s, padding='same'):
        return Conv2D(
            filters,
            k,
            strides=(s, s),
            padding=padding,
            activation=tf.nn.relu6,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
            )

    def maxpool2d(self, ps, s, padding='same'):
        return MaxPooling2D(
            pool_size=(ps, ps),
            strides=(s, s),
            padding=padding
            )

    def dense(self, units):
        return Dense(
            units,
            activation=tf.nn.relu6,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
            )

    """
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

    def label_heatmap(self, land):
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

    def img_heatmap(self, land):
        # land: [68, 2]
        one_label_heatmap = tf.map_fn(self.label_heatmap, land)
        labels_heatmap = tf.reduce_sum(one_label_heatmap, 0, name="label_heatmap")

        return labels_heatmap

    def HeatMap(self, output):
        # ?*136 / transform / ?*112*112*1
        # import pdb;pdb.set_trace()
        # landmark: [?, 68, 2]
        landmark = tf.reshape(output, [-1, 68, 2])
        heatmap = tf.map_fn(self.img_heatmap, landmark)
        heatmap = tf.expand_dims(heatmap, -1, name="heatmap")
        print(heatmap.name, heatmap.shape)

        return heatmap, [landmark, heatmap]

    def call(self, input):
        print("=== start network ===")
        # import pdb;pdb.set_trace()
        print('PFLD input shape({}): {}'.format(input.name, input.shape))
        ###### stage1 ######
        print("=== start stage 1 ===")
        # 112*112*3(1) / conv3*3 / c:16,n:1,s:2
        _conv1_1 = self.conv1_1(input)
        print(_conv1_1.name, _conv1_1.shape)
        # 56*56*16 / conv3*3 / c:32,n:1,s:2
        _conv1_2 = self.conv1_2(_conv1_1)
        print(_conv1_2.name, _conv1_2.shape)
        # 28*28*32 / pool2*2 / c:32,n:1,s:2
        _pool1_2 = self.pool1_2(_conv1_2)
        print(_pool1_2.name, _pool1_2.shape)
        # 14*14*32 / conv3*3 / c:64,n:1,s:2
        _conv1_2_1 = self.conv1_2_1(_pool1_2)
        print(_conv1_2_1.name, _conv1_2_1.shape)
        # 7*7*64 / global_pool / c:64,n:1
        _pool1_2_1 = self.pool1_2_1(_conv1_2_1)
        print(_pool1_2_1.name, _pool1_2_1.shape)
        # 14*14*32 / conv3*3 / c:64,n:1,s:2
        _conv1_3 = self.conv1_3(_pool1_2)
        print(_conv1_3.name, _conv1_3.shape)
        # 7*7*64 / pool2*2 / c:64,n:1,s:2
        _pool1_3 = self.pool1_3(_conv1_3)
        print(_pool1_3.name, _pool1_3.shape)
        # 4*4*64 / conv3*3 / c:64,n:1,s:2
        _conv1_3_1 = self.conv1_3_1(_pool1_3)
        print(_conv1_3_1.name, _conv1_3_1.shape)
        # 2*2*64 / global_pool / c:64,n:1
        _pool1_3_1 = self.pool1_3_1(_conv1_3_1)
        print(_pool1_3_1.name, _pool1_3_1.shape)
        # 4*4*64 / conv3*3 / c:64,n:1,s:2
        _conv1_4 = self.conv1_4(_pool1_3)
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

        ###### stage2 ######
        print("=== start stage 2 ===")
        # 112*112*1*2 / concat / 112*112*2
        _concatted_2 = tf.concat([input, _heatmap], 3)
        print(_concatted_2.name, _concatted_2.shape)
        # 112*112*2 / conv3*3 / c:8,n:1,s:2
        _conv2_1 = self.conv2_1(_concatted_2)
        print(_conv2_1.name, _conv2_1.shape)
        # 56*56*8 / pool3*3 / c:28,n:1,s:2
        _pool2_1 = self.pool2_1(_conv2_1)
        print(_pool2_1.name, _pool2_1.shape)
        # 28*28*8 / conv3*3 / c:16,n:1,s:1
        _conv2_2 = self.conv2_2(_pool2_1)
        print(_conv2_2.name, _conv2_2.shape)
        # 28*28*16 / pool3*3 / c:16,n:1,s:2
        _pool2_2 = self.pool2_2(_conv2_2)
        print(_pool2_2.name, _pool2_2.shape)
        # 14*14*16 / conv3*3 / c:64,n:1,s:1
        _conv2_2_1 = self.conv2_2_1(_pool2_2)
        print(_conv2_2_1.name, _conv2_2_1.shape)
        # 14*14*64 / global_pool / c:64,n:1
        _pool2_2_1 = self.pool2_2_1(_conv2_2_1)
        print(_pool2_2_1.name, _pool2_2_1.shape)
        # 14*14*16 / conv3*3 / c:64,n:1,s:2
        _conv2_3 = self.conv2_3(_pool2_2)
        print(_conv2_3.name, _conv2_3.shape)
        # 7*7*64 / pool3*3 / c:64,n:1,s:2
        _pool2_3 = self.pool2_3(_conv2_3)
        print(_pool2_3.name, _pool2_3.shape)
        # 3*3*64 / conv3*3 / c:64,n:1,s:1
        _conv2_3_1 = self.conv2_3_1(_pool2_3)
        print(_conv2_3_1.name, _conv2_3_1.shape)
        # 3*3*64 / global_pool / c:64,n:1
        _pool2_3_1 = self.pool2_3_1(_conv2_3_1)
        print(_pool2_3_1.name, _pool2_3_1.shape)
        # 3*3*64 / conv3*3 / c:64,n:1,s:2
        _conv2_4 = self.conv2_4(_pool2_3)
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

        return _output_2, _heat_values


def sample():
    pass
    """
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
    """

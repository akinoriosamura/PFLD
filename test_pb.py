from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import gfile

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
import cv2

from generate_data import gen_data


class Model(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath

    def adjust_graph(self, graph_def):

        # fix nodes
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        return graph_def

    def load_graph(self, frozen_graph_filename):
        # pbファイルを読み込みgraph定義を復元する
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph = self.graph)

        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            graph_def = self.adjust_graph(graph_def)

        image_size = 112
        self.image_batch = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3),\
                                name='image_batch')
        self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        tf.import_graph_def(graph_def, {'image_batch': self.image_batch, 'phase_train': self.phase_train_placeholder})

        print('Model loading complete!')

        # self.print_graph_nodes(graph_def)


    def print_graph_operations(self, graph):
        # print operations
        print("----- operations in graph -----")
        for op in graph.get_operations():
            print(op.name,op.outputs)

    def print_graph_nodes(self, graph_def):
        # print nodes
        print("----- nodes in graph_def -----")
        for node in graph_def.node:
            print(node)

    def test(self):

        image_size = 112

        image_files = 'data/test_original_data/list_sample.txt'
        out_dir = 'sample_test_result'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        self.load_graph(self.model_filepath)
        # print operations
        # self.print_graph_operations(self.graph)
        landmark_total = self.graph.get_tensor_by_name('import/pfld_inference/fc/BiasAdd:0')
        
        print('Loading feature extraction model.')

        file_list, train_landmarks, train_attributes, euler = gen_data(image_files)
        print(file_list)
        for file in file_list:
            filename = os.path.split(file)[-1]
            image = cv2.imread(file)
            # image = cv2.resize(image, (image_size, image_size))
            input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            input = cv2.resize(input, (image_size, image_size))
            input = input.astype(np.float32)/256.0
            input = np.expand_dims(input, 0)
            print(input.shape)

            feed_dict = {
                self.image_batch: input,
                self.phase_train_placeholder: False
            }

            pre_landmarks = self.sess.run(landmark_total, feed_dict = feed_dict)

            print(pre_landmarks)
            pre_landmark = pre_landmarks[0]

            h, w, _ = image.shape
            pre_landmark = pre_landmark.reshape(-1, 2) * [h, w]
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(image, (x, y), 1, (0, 0, 255))
            cv2.imwrite(os.path.join(out_dir, filename), image)


if __name__ == "__main__":
    pb_path = "./original_98_frozen.pb"
    model = Model(pb_path)
    model.test()
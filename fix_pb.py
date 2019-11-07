from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf


class Model(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath

    def adjust_graph(self, graph_def):

        # fix nodes
        """
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
        """
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
                if 'validate_shape' in node.attr:
                    del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    node.input[0] = node.input[1]
                    del node.input[1]

        return graph_def

    def test(self):
        with tf.Graph().as_default() as graph:
            sess = tf.InteractiveSession(graph=graph)
            import pdb; pdb.set_trace()
            with tf.gfile.GFile(self.model_filepath, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

                graph_def = self.adjust_graph(graph_def)

            image_size = 112
            self.image_batch = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3),
                                              name='image_batch')
            self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            tf.import_graph_def(
                graph_def,
                {'image_batch': self.image_batch, 'phase_train': self.phase_train_placeholder}
            )
            print('Model loading complete!')

            # save graphdef file to pb
            # gd = sess.graph.as_graph_def()
            print("Save frozen graph")
            graphdef_n = "fixed_original_98_frozen.pb"
            fixed_graph_def = graph_util.convert_variables_to_constants(
                sess, graph_def, ["pfld_inference/fc/BiasAdd"])
            tf.train.write_graph(fixed_graph_def, "./", graphdef_n, as_text=False)
            print("finish saving")


if __name__ == "__main__":
    pb_path = "./original_98_frozen.pb"
    model = Model(pb_path)
    model.test()

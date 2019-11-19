from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import tensorflow as tf
import numpy as np
import cv2
import shutil
import time

from generate_data import gen_data


def main():
    num_labels = 68
    saved_target = "./models2/save_models/pcn_growing_68/1113/"
    meta_file = saved_target + 'model.meta'
    ckpt_file = saved_target + 'model.ckpt-89'
    # test_list = './data/300w_image_list.txt'

    image_size = 112

    image_files = 'data/test_WFLW_68_data/list.txt'
    out_dir = 'sample_WFLW_test_result'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            print('Loading feature extraction model.')
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(tf.get_default_session(), ckpt_file)

            graph = tf.get_default_graph()
            images_placeholder = graph.get_tensor_by_name('image_batch:0')
            phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')

            """
            landmark_L1 = graph.get_tensor_by_name('landmark_L1:0')
            landmark_L2 = graph.get_tensor_by_name('landmark_L2:0')
            landmark_L3 = graph.get_tensor_by_name('landmark_L3:0')
            landmark_L4 = graph.get_tensor_by_name('landmark_L4:0')
            landmark_L5 = graph.get_tensor_by_name('landmark_L5:0')
            landmark_total = [landmark_L1, landmark_L2, landmark_L3, landmark_L4, landmark_L5]
            """
            landmark_total = graph.get_tensor_by_name(
                'pfld_inference/fc/BiasAdd:0')

            file_list, train_landmarks, train_attributes, euler_angles = gen_data(
                image_files, num_labels)
            print(file_list)
            for file in file_list:
                filename = os.path.split(file)[-1]
                image = cv2.imread(file)
                # image = cv2.resize(image, (image_size, image_size))
                input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                input = cv2.resize(input, (image_size, image_size))
                input = input.astype(np.float32) / 256.0
                input = np.expand_dims(input, 0)
                # print(input.shape)

                feed_dict = {
                    images_placeholder: input,
                    phase_train_placeholder: False
                }

                st = time.time()
                pre_landmarks = sess.run(landmark_total, feed_dict=feed_dict)
                # print(pre_landmarks)
                print("elaps: ", time.time() - st)
                pre_landmark = pre_landmarks[0]

                h, w, _ = image.shape
                pre_landmark = pre_landmark.reshape(-1, 2) * [h, w]
                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(image, (x, y), 1, (0, 0, 255))
                print(os.path.join(out_dir, filename))
                cv2.imwrite(os.path.join(out_dir, filename), image)


if __name__ == '__main__':
    main()

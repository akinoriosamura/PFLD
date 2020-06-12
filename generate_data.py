import tensorflow as tf
import numpy as np
import cv2
import time
import os


class DataLoader():
    def __init__(self, file_list, args, phase, debug=False):
        print("labels; ", args.num_labels)
        time.sleep(3)
        self.file_list = file_list
        self.args = args
        self.phase = phase
        self.file_list, self.landmarks, self.attributes, self.euler_angles = self.gen_data(self.file_list, self.args.num_labels)
        self.num_file = len(self.file_list)

        if debug:
            n = self.args.batch_size * 10
            file_list = self.file_list[:n]
            landmarks = self.landmarks[:n]
            attributes = self.attributes[:n]
            euler_angles = self.euler_angles[:n]

    def gen_data(self, file_list, num_labels):
        with open(file_list, 'r') as f:
            lines = f.readlines()
        filenames, landmarks, attributes, euler_angles = [], [], [], []
        for line in lines:
            line = line.strip().split()
            path = line[0]
            landmark = line[1:num_labels*2+1]  # 1:197
            attribute = line[num_labels*2+1:num_labels*2+7]  # 197:203
            euler_angle = line[num_labels*2+7:num_labels*2+10]  # 203:206

            landmark = np.asarray(landmark, dtype=np.float32)
            attribute = np.asarray(attribute, dtype=np.int32)
            euler_angle = np.asarray(euler_angle, dtype=np.float32)
            filenames.append(path)
            landmarks.append(landmark)
            attributes.append(attribute)
            euler_angles.append(euler_angle)
        filenames = np.asarray(filenames, dtype=np.str)
        landmarks = np.asarray(landmarks, dtype=np.float32)
        attributes = np.asarray(attributes, dtype=np.int32)
        euler_angles = np.asarray(euler_angles, dtype=np.float32)
        return (filenames, landmarks, attributes, euler_angles)

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.file_list, self.landmarks, self.attributes, self.euler_angles))

        def _parse_data(filename, landmarks, attributes, euler_angles):
            # filename, landmarks, attributes = data
            file_contents = tf.read_file(filename)
            image = tf.image.decode_png(file_contents, channels=self.args.image_channels)
            # print(image.get_shape())
            # image.set_shape((args.image_size, args.image_size, args.image_channels))
            image = tf.image.resize_images(image, (self.args.image_size, self.args.image_size), method=0)
            image = tf.cast(image, tf.float32)

            image = image / 256.0
            return (image, landmarks, attributes, euler_angles)
        
        dataset = dataset.map(_parse_data)
        dataset = dataset.shuffle(buffer_size=10000)
        return dataset, self.num_file


if __name__ == '__main__':
    file_list = 'data/train_data/list.txt'
    num_labels = 98
    filenames, landmarks, attributes = gen_data(file_list, num_labels)
    for i in range(len(filenames)):
        filename = filenames[i]
        landmark = landmarks[i]
        attribute = attributes[i]
        print(attribute)
        img = cv2.imread(filename)
        h, w, _ = img.shape
        landmark = landmark.reshape(-1, 2) * [h, w]
        for (x, y) in landmark.astype(np.int32):
            cv2.circle(img, (x, y), 1, (0, 0, 255))
        cv2.imshow('0', img)
        cv2.waitKey(0)

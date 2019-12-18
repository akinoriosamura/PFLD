import tensorflow as tf
import numpy as np
import cv2
import time
import os

from data_augmentor import DataAugmentator

class DataLoader():
    def __init__(self, file_list, args, phase, debug=False):
        print("labels; ", args.num_labels)
        time.sleep(3)
        self.file_list = file_list
        self.args = args
        self.phase = phase
        self.file_list, self.landmarks, self.attributes, self.euler_angles = self.gen_data(self.file_list, self.args.num_labels)
        self.images_shape = None
        self.landmarks_shape = list(self.landmarks.shape)
        self.attributes_shape = list(self.attributes.shape)
        self.euler_angles_shape = list(self.euler_angles.shape)

        if debug:
            n = self.args.batch_size * 10
            file_list = self.file_list[:n]
            landmarks = self.landmarks[:n]
            attributes = self.attributes[:n]
            euler_angles = self.euler_angles[:n]
        self.images = None
        self.num_file = len(self.file_list)
        self.dataaugmentor = DataAugmentator(self.args.num_labels)


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

    def make_example(self, image, landmark, attribute, euler_angle):
        return tf.train.Example(features=tf.train.Features(feature={
            'image' : tf.train.Feature(float_list=tf.train.FloatList(value=image.reshape(-1))),
            'landmark' : tf.train.Feature(float_list=tf.train.FloatList(value=landmark.reshape(-1))),
            'attribute' : tf.train.Feature(float_list=tf.train.FloatList(value=attribute.reshape(-1))),
            'euler_angle' : tf.train.Feature(float_list=tf.train.FloatList(value=euler_angle.reshape(-1)))
        }))

    def write_tfrecord(self, tfrecord_path): 
        writer = tf.io.TFRecordWriter(tfrecord_path)
        for image, landmark, attribute, euler_angle in zip(self.images, self.landmarks, self.attributes, self.euler_angles):
            ex = self.make_example(image, landmark, attribute, euler_angle)
            writer.write(ex.SerializeToString())
        writer.close()

    def parse_function(self, example_proto):
        features = {"image": tf.FixedLenFeature(self.images_shape[1:], tf.float32),
                "landmark": tf.FixedLenFeature(self.landmarks_shape[1:], tf.float32),
                "attribute": tf.FixedLenFeature(self.attributes_shape[1:], tf.float32),
                "euler_angle": tf.FixedLenFeature(self.euler_angles_shape[1:], tf.float32)}
    
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features['image'], parsed_features['landmark'], parsed_features['attribute'], parsed_features['euler_angle']

    def gen_tfrecord(self):
        def _parse_data(fname):
            image = cv2.imread(fname)
            image = cv2.resize(image, (self.args.image_size, self.args.image_size))

            return image

        def _normalize(image):
            image = image.astype(np.float32)
            image = image / 256.0

            return image
            
        self.images = np.array(list(map(_parse_data, self.file_list)))
        self.images_shape = list(self.images.shape)
        #o_images = self.images
        #o_landmarks = self.landmarks
        if self.phase == "train" and self.args.is_augment:
            print("======= augment =========")
            tfrecord_path = "./data/augment_" + self.phase + ".tfrecords"
            augments = np.array(list(map(self.dataaugmentor.augment_image, self.images, self.landmarks)))
            self.images = np.array(list(augments[:, 0]))
            self.landmarks = np.array(list(augments[:, 1]))
        else:
            print("======= not augment =========")
            tfrecord_path = "./data/unaugment_" + self.phase + ".tfrecords"
        #import pdb;pdb.set_trace()
        #save_anno(o_images[0], o_landmarks[0]*256, "ori")
        #save_anno(self.images[0], (self.landmarks[0]*256), "aug")

        self.images = np.array(list(map(_normalize, self.images)))
        self.write_tfrecord(tfrecord_path)
        print("save in record dataset : ", tfrecord_path)

        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(self.parse_function)
        dataset = dataset.shuffle(buffer_size=10000)

        return dataset


def save_anno(img, lands, type):
    lands = lands.astype(np.int32)
    for id in range(int(len(lands) / 2), 2):
        x = lands[id]
        y = lands[id+1]
        cv2.circle(img, (x, y), 1, (0, 0, 255))
    cv2.imwrite("./img_" + type + ".jpg", img)

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

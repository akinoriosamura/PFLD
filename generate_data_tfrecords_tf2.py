import tensorflow as tf
import numpy as np
import math
import cv2
import random
import time
import os

from data_augmentor import DataAugmentator


class TfrecordsLoader():
    def __init__(self, file_list, args, phase, model_type, debug=False):
        print("labels; ", args.num_labels)
        time.sleep(3)
        self.num_records = 2
        self.file_list = file_list
        with open(self.file_list, 'r') as f:
            self.lines = f.readlines()
        self.num_file = len(self.lines)
        self.file_base = os.path.basename(os.path.dirname(self.file_list))
        self.args = args
        self.tfrecords_dir = os.path.join(
            self.args.tfrecords_dir, self.file_base)
        self.dataaugmentor = DataAugmentator(self.args.num_labels)
        self.phase = phase
        self.model_type = model_type
        self.sumShape = None
        self.meanShape = None
        self.filenames = None
        self.landmarks = None
        self.attributes = None
        self.euler_angles = None
        line = self.lines[0].strip().split()
        landmark = line[1:self.args.num_labels*2+1]  # 1:197
        attribute = line[self.args.num_labels*2 +
                         1:self.args.num_labels*2+7]  # 197:203
        euler_angle = line[self.args.num_labels*2 +
                           7:self.args.num_labels*2+10]  # 203:206
        landmark = np.asarray(landmark, dtype=np.float32)
        attribute = np.asarray(attribute, dtype=np.int32)
        euler_angle = np.asarray(euler_angle, dtype=np.float32)
        self.images_shape = [self.args.image_size, self.args.image_size, 3]
        self.landmarks_shape = landmark.shape
        self.attributes_shape = attribute.shape
        self.euler_angles_shape = euler_angle.shape
        self.images = None
        self.records_list = []
        self.meanShape = None

    def set_data(self, _lines):
        filenames, landmarks, attributes, euler_angles = [], [], [], []
        for line in _lines:
            line = line.strip().split()
            path = line[0]
            landmark = line[1:self.args.num_labels*2+1]  # 1:197
            attribute = line[self.args.num_labels*2 +
                             1:self.args.num_labels*2+7]  # 197:203
            euler_angle = line[self.args.num_labels*2 +
                               7:self.args.num_labels*2+10]  # 203:206

            landmark = np.asarray(landmark, dtype=np.float32)
            attribute = np.asarray(attribute, dtype=np.int32)
            euler_angle = np.asarray(euler_angle, dtype=np.float32)
            filenames.append(path)
            landmarks.append(landmark)
            attributes.append(attribute)
            euler_angles.append(euler_angle)
        self.filenames = np.asarray(filenames, dtype=np.str)
        if self.model_type == 'pfld':
            print(" get normalized landmark in pfld")
            self.landmarks = np.asarray(landmarks, dtype=np.float32)
        elif self.model_type == 'xin':
            print(" get scale landmark in xin")
            self.landmarks = np.asarray(
                landmarks, dtype=np.float32)  # * self.args.image_size
            lands = self.landmarks.copy()
            for land in lands:
                if self.sumShape is None:
                    self.sumShape = land
                else:
                    self.sumShape += land
        self.attributes = np.asarray(attributes, dtype=np.int32)
        self.euler_angles = np.asarray(euler_angles, dtype=np.float32)

    def make_example(self, image, landmark, attribute, euler_angle):
        return tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(float_list=tf.train.FloatList(value=image.reshape(-1))),
            'landmark': tf.train.Feature(float_list=tf.train.FloatList(value=landmark.reshape(-1))),
            'attribute': tf.train.Feature(float_list=tf.train.FloatList(value=attribute.reshape(-1))),
            'euler_angle': tf.train.Feature(float_list=tf.train.FloatList(value=euler_angle.reshape(-1)))
        }))

    def write_one_tfrecord(self, tfrecord_path):
        writer = tf.io.TFRecordWriter(tfrecord_path)
        for image, landmark, attribute, euler_angle in zip(self.images, self.landmarks, self.attributes, self.euler_angles):
            ex = self.make_example(image, landmark, attribute, euler_angle)
            writer.write(ex.SerializeToString())
        writer.close()

    def write_tfrecord(self, tfrecord_base_path):
        def _parse_data(fname):
            image = cv2.imread(fname)
            image = cv2.resize(
                image, (self.args.image_size, self.args.image_size))
            return image

        def _normalize(image):
            image = image.astype(np.float32)
            if self.model_type == 'pfld':
                # print("normalizetion: / 256 in pfld")
                image = image / 256.0
            elif self.model_type == 'xin':
                # print("normalizetion: -127.5 / 127.5 in xin")
                image = (image - 127.5) / 127.5
            return image

        self.images = np.array(list(map(_parse_data, self.filenames)))
        print("loaded daatset in np")
        o_images = self.images
        o_landmarks = self.landmarks
        # import pdb;pdb.set_trace()
        if self.phase == "train" and self.args.is_augment:
            print("======= augment =========")
            tfrecord_path = os.path.join(
                self.tfrecords_dir, ("augment_" + tfrecord_base_path))
            augments = np.array(
                list(map(self.dataaugmentor.augment_image, self.images, self.landmarks)))
            self.images = np.array(list(augments[:, 0]))
            self.landmarks = np.array(list(augments[:, 1]))
        else:
            print("======= not augment =========")
            tfrecord_path = os.path.join(
                self.tfrecords_dir, ("unaugment_" + tfrecord_base_path))
        self.save_anno(o_images, o_landmarks, "test")
        #self.save_anno(self.images, (self.landmarks), "aug")

        self.images = np.array(list(map(_normalize, self.images)))
        self.write_one_tfrecord(tfrecord_path)
        print("save in record dataset : ", tfrecord_path)

        return tfrecord_path

    def create_tfrecord(self):
        if os.path.exists(self.tfrecords_dir):
            print("tfrecords exist")
            records_files = os.listdir(self.tfrecords_dir)
            for record_file in records_files:
                if self.phase == 'train':
                    if "train.tfrecords" in record_file:
                        print("append record files: ", os.path.join(
                            self.tfrecords_dir, record_file))
                        self.records_list.append(os.path.join(
                            self.tfrecords_dir, record_file))
                elif self.phase == 'test':
                    if "test.tfrecords" in record_file:
                        print("append record files: ", os.path.join(
                            self.tfrecords_dir, record_file))
                        self.records_list.append(os.path.join(
                            self.tfrecords_dir, record_file))
        else:
            print("tfrecords no exist")
            os.makedirs(self.tfrecords_dir, exist_ok=True)
            if self.phase == 'train':
                num_record_files = math.ceil(self.num_file / self.num_records)
                for id_record in range(0, self.num_records):
                    i_st = id_record * num_record_files
                    if i_st + num_record_files < self.num_file:
                        i_end = i_st + num_record_files
                    else:
                        i_end = self.num_file
                    print("index: {0} : {1}".format(i_st, i_end))
                    part_lines = self.lines[i_st: i_end]
                    self.set_data(part_lines)
                    tfrecord_base_path = self.phase + \
                        ".tfrecords." + str(id_record)
                    tfrecord_path = self.write_tfrecord(tfrecord_base_path)
                    self.records_list.append(tfrecord_path)
            elif self.phase == 'test':
                self.set_data(self.lines)
                tfrecord_base_path = self.phase + ".tfrecords"
                tfrecord_path = self.write_tfrecord(tfrecord_base_path)
                self.records_list.append(tfrecord_path)

        print("records_list: ",  self.records_list)

    def parse_function(self, example_proto):
        features = {"image": tf.io.FixedLenFeature(self.images_shape, tf.float32),
                    "landmark": tf.io.FixedLenFeature(self.landmarks_shape, tf.float32),
                    "attribute": tf.io.FixedLenFeature(self.attributes_shape, tf.float32),
                    "euler_angle": tf.io.FixedLenFeature(self.euler_angles_shape, tf.float32)}

        parsed_features = tf.io.parse_single_example(example_proto, features)

        return parsed_features['image'], parsed_features['landmark'], parsed_features['attribute'], parsed_features['euler_angle']

    def get_tfrecords(self, tfrecord_path):
        # https://datascience.stackexchange.com/questions/16318/what-is-the-benefit-of-splitting-tfrecord-file-into-shards
        print("get from : ", tfrecord_path)
        dataset = tf.data.TFRecordDataset(
            tfrecord_path,
            num_parallel_reads=os.cpu_count()
        )
        dataset = dataset.map(self.parse_function)
        dataset = dataset.shuffle(buffer_size=int(
            self.num_file / (self.num_records)))
        print("get tfrecords")

        return dataset

    def calMeanShape(self):
        mf_path = os.path.join(self.tfrecords_dir, "mean_face_shape.txt")
        if os.path.exists(mf_path):
            with open(mf_path, mode='r') as mf:
                _meanShape = mf.readline()
                _meanShape = _meanShape.split(" ")
                self.meanShape = np.array([float(mf) for mf in _meanShape])
        else:
            self.meanShape = (self.sumShape / self.num_file)
            # save mean shape txt
            with open(mf_path, mode='w') as mf:
                ms_str_list = [str(ms) for ms in self.meanShape.tolist()]
                ms_str = " ".join(ms_str_list)
                mf.write(ms_str)
            # save mean face demo
            lands = self.meanShape.reshape(-1, 2) * self.args.image_size
            lands = np.asarray(lands, np.int32)
            img_tmp = self.images[10].copy()
            # print("normalizetion: -127.5 / 127.5 in xin")
            img_tmp = (img_tmp * 127.5) + 127.5
            # img_tmp = (img_tmp * 256)
            for land_id, (x, y) in enumerate(lands):
                cv2.circle(img_tmp, (x, y), 1, (0, 255, 0))
                # cv2.putText(img_tmp, str(land_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), thickness=1)
            cv2.imwrite("./img_mean_shape.jpg", img_tmp)

    def save_anno(self, imgs, landmarks, type):
        # import pdb;pdb.set_trace()
        for i in range(10):
            img = imgs[i].copy()
            lands = landmarks[i].copy()
            h, w, _ = img.shape
            lands = lands.reshape(-1, 2)
            if self.model_type == 'pfld':
                # print(" scale: h, w in save anno in pfld")
                lands = np.asarray(lands * [h, w], np.int32)
            elif self.model_type == 'xin':
                # print(" no scale in save anno in xin")
                lands = np.asarray(lands * [h, w], np.int32)
                # lands = np.asarray(lands, np.int32)
            for land_id, (x, y) in enumerate(lands):
                cv2.circle(img, (x, y), 1, (0, 255, 0))
                # cv2.putText(img, str(land_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), thickness=1)
            cv2.imwrite("./img_" + str(i) + type + ".jpg", img)
        print("finish save anno")


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

import tensorflow as tf
from generate_data_tfrecords_tf2 import TfrecordsLoader
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Model
tf.keras.backend.set_floatx('float32')
# from pfld import create_model
from XinNing2020_tf2 import XinNingNetwork

import os
import sys
import gc
import time
import random
import math
import numpy as np
import cv2
import argparse
import shutil


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d0 = Conv2D(16, 3, padding='valid', activation='relu', 
          input_shape=(112, 112 ,3), kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.d1 = MaxPooling2D()
        self.d2 = Conv2D(16, 3, padding='valid', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.d3 = MaxPooling2D()
        self.flat = Flatten()
        self.d4_1 = Dense(136, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.d4_2 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))

    def call(self, x):
        x = self.d0(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.flat(x)
        y_1 = self.d4_1(x)
        # print(y_1.shape)
        # y_2 = self.d4_2(x)
        # print(y_2.shape)
        return y_1# , y_2


def main(args):
  debug = (args.debug == 'True')
  print("args: ", args)
  np.random.seed(args.seed)
  time.sleep(3)

  model_dir = args.model_dir
  print('Model dir: {}'.format(model_dir))
  os.makedirs(model_dir, exist_ok=True)

  # ============== get dataset ==============
  # use tfrecord
  print("============== get dataloader ==============")
  train_loader = TfrecordsLoader(args.file_list, args, "train", "pfld")
  test_loader = TfrecordsLoader(args.test_list, args, "test", "pfld")
  print("============ get tfrecord train data ===============")
  train_loader.create_tfrecord()
  num_train_file = train_loader.num_file
  print("============ get tfrecord test data ===============")
  test_loader.create_tfrecord()
  num_test_file = test_loader.num_file

  # run below in epoch for large dataset
  test_dataset = test_loader.get_tfrecords(test_loader.records_list[0])
  batch_test_dataset = test_dataset.batch(args.batch_size)

  print('Total number of examples: {}'.format(num_train_file))
  print('Test number of examples: {}'.format(num_test_file))
  epoch_size = 1 + (num_train_file // args.batch_size)
  print('Number of batches per epoch: {}'.format(epoch_size))

  # ================== create models ================
  print("=================== create models ===============")
  model = XinNingNetwork(args.num_labels, args.image_size)
  # print(model.summary())

  def loss_objects(outputs, targets, L2_losses):
    landmarks_pre = outputs[0]
    landmark_batch, euler_batch = targets[0], targets[1]
    # attributes_w_n = tf.to_float(attribute_batch[:, 1:6])
    # _num = attributes_w_n.shape[0]
    # mat_ratio = tf.reduce_mean(attributes_w_n, axis=0)
    # mat_ratio = tf.map_fn(lambda x: (tf.cond(x > 0, lambda: 1 / x, lambda: float(args.batch_size))), mat_ratio)
    # attributes_w_n = tf.convert_to_tensor(attributes_w_n * mat_ratio)
    # attributes_w_n = tf.reduce_sum(attributes_w_n, axis=1)

    L2_loss = tf.add_n(L2_losses)
    # _sum_k = tf.reduce_sum(tf.map_fn(lambda x: 1 - tf.cos(abs(x)), euler_angles_gt_batch - euler_angles_pre), axis=1)
    loss_sum = tf.reduce_sum(tf.square(landmark_batch - landmarks_pre), axis=1)
    loss_sum = tf.reduce_mean(loss_sum)#  * _sum_k)#  * attributes_w_n)
    loss_sum += L2_loss

    return loss_sum, L2_loss

  boundaries = [int(bound) * epoch_size for bound in args.lr_epoch.split(',')]
  lr_sc = [args.learning_rate * (0.1 ** b_i) for b_i in range(len(boundaries))]
  lr_sc.append(0.00000001)
  learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries, lr_sc)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn(0))

  @tf.function
  def train_step(inputs, targets):
      with tf.GradientTape() as tape:
          outputs = model(inputs)
          loss_value, L2_loss = loss_objects(outputs, targets, model.losses)

      gradients = tape.gradient(loss_value, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      return outputs, loss_value, L2_loss

  epoch_start = 0

  # ============ resotre pretrain =============
  print("================= resotre pretrain if exist =================")
  if args.pretrained_model:
      pretrained_model = args.pretrained_model
      root = tf.train.Checkpoint(optimizer=optimizer, model=model)
      root.restore(tf.train.latest_checkpoint(pretrained_model))
      print('Restore from model directory: {}'.format(pretrained_model))

  all_step = 0
  for epoch in range(epoch_start, args.max_epoch):
    batch_num = 0
    print("get dataset start")
    records_order = random.sample(train_loader.records_list, train_loader.num_records)
    assert len(records_order) == train_loader.num_records
    print("records order: ", records_order)
    for record_id, target_train_tfrecord_path in enumerate(records_order):
      if record_id != 0:
        print("delete dataset memory ")
        del train_dataset
        del batch_train_dataset
        gc.collect()
      print("target_train_tfrecord_path : ", target_train_tfrecord_path)
      train_dataset = train_loader.get_tfrecords(target_train_tfrecord_path)
      batch_train_dataset = train_dataset.batch(args.batch_size)

      for batch_i, (image_batch, landmarks_batch, _, euler_batch) in enumerate(batch_train_dataset):
          outs, losses, L2_loss = train_step(image_batch, [landmarks_batch, euler_batch])
          if ((batch_i + 1) % 10) == 0 or (batch_i + 1) == epoch_size:
              Epoch = 'Epoch:[{:<4}][{:<4}/{:<4}][{:<4}/{:<4}]'.format(epoch, record_id + 1, train_loader.num_records, batch_num, epoch_size)
              # import pdb;pdb.set_trace()
              Loss = 'Loss {:2.3f} L2 loss {:2.3f}'.format(losses, L2_loss), 
              print('{}\t{}\t lr {:2.3}'.format(Epoch, Loss, optimizer.learning_rate))
          batch_num += 1
          all_step += 1
          optimizer.learning_rate = learning_rate_fn(all_step)

    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
    root = tf.train.Checkpoint(optimizer=optimizer, model=model)
    root.save(checkpoint_path)
    savedmodel_path = os.path.join(model_dir, 'SavedModel/')
    tf.saved_model.save(model, savedmodel_path)
    print("save checkpoint: {}".format(checkpoint_path))
    print("save SavedModel: {}".format(savedmodel_path))

    if epoch % 1 == 0 and epoch != 0 and epoch > 0:
        print("test start")
        start = time.time()
        test(batch_test_dataset, num_test_file, model, args)
        print("test time: {}" .format(time.time() - start))

def test(batch_test_dataset, num_test_file, model, args):
  loss_sum = 0
  landmark_error = 0
  landmark_01_num = 0

  epoch_size = math.ceil(num_test_file * 1.0 / args.batch_size)
  print("num test file: ", num_test_file)
  print("test epoch size: ", epoch_size)
  for i, (image_batch, landmarks_batch, attribute_batch, euler_batch) in enumerate(batch_test_dataset):  # batch_num
      print("start epoch: ", i)
      pre_landmarks = model(image_batch)

      diff = pre_landmarks - landmarks_batch
      loss = np.sum(diff * diff)
      loss_sum += loss

      for k in range(pre_landmarks.shape[0]):
          error_all_points = 0
          for count_point in range(pre_landmarks.shape[1] // 2):  # num points
              error_diff = pre_landmarks[k][(count_point * 2):(count_point * 2 + 2)] - \
                  landmarks_batch[k][(count_point * 2):(count_point * 2 + 2)]
              error = np.sqrt(np.sum(error_diff * error_diff))
              error_all_points += error
          if (args.num_labels == 52) or (args.num_labels == 68) or (args.num_labels == 98):
              # 目の両端
              if args.num_labels == 98:
                  left_eye_edge = 60
                  right_eye_edge = 72
              elif args.num_labels == 68:
                  left_eye_edge = 36
                  right_eye_edge = 45
              elif args.num_labels == 52:
                  left_eye_edge = 20
                  right_eye_edge = 29
              else:
                  print("eye error")
                  exit()
              time.sleep(3)
              interocular_distance = np.sqrt(
                  np.sum(
                      pow((landmarks_batch[k][left_eye_edge*2:left_eye_edge*2+2] - landmarks_batch[k][right_eye_edge*2:right_eye_edge*2+2]), 2)
                      )
              )
              error_norm = error_all_points / (interocular_distance * args.num_labels)
          else:
              error_norm = error_all_points
          landmark_error += error_norm
          if error_norm >= 0.02:
              landmark_01_num += 1

  loss = loss_sum / (num_test_file * 1.0)
  print('Test epochs: {}\tLoss {:2.3f}'.format(epoch_size, loss))

  print('mean error and failure rate')
  landmark_error_norm = landmark_error / (num_test_file * 1.0)
  error_str = 'mean error : {:2.3f}'.format(landmark_error_norm)

  failure_rate_norm = landmark_01_num / (num_test_file * 1.0)
  print("landmark_01_num: ", landmark_01_num)
  failure_rate_str = 'failure rate: L1 {:2.3f}'.format(failure_rate_norm)
  print(error_str + '\n' + failure_rate_str + '\n')


def parse_arguments(argv):
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_list', type=str, default='data/train_data/list.txt')
    parser.add_argument('--test_list', type=str, default='data/test_data/list.txt')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--max_epoch', type=int, default=10000)
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--num_labels', type=int, default=98)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default='models1/model_test')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_epoch', type=str, default='20,30,40,60,100,160,180,200,500,990,1010')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--level', type=str, default='L5')
    parser.add_argument('--save_image_example', action='store_false')
    parser.add_argument('--debug', type=str, default='False')
    parser.add_argument('--depth_multi', type=float, default=1)
    parser.add_argument('--num_quant', type=int, default=64)
    parser.add_argument('--tfrecords_dir', type=str, default='/data/tfrecords_xin')
    parser.add_argument('--is_augment', type=str2bool, default=False, help='Whether to augment')

    return parser.parse_args(argv)


if __name__ == '__main__':
    print(sys.argv)
    main(parse_arguments(sys.argv[1:]))

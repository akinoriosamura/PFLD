import shutil
import argparse
import cv2
import numpy as np
import math
import random
import time
import gc
import sys
import os
from XinNing2020_tf2 import XinNingNetwork
import tensorflow as tf
from generate_data_tfrecords_tf2 import TfrecordsLoader
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
tf.keras.backend.set_floatx('float32')
# from pfld import create_model


def main(args):
    train_stage = 'stage1'
    print("============= this phase is : ", train_stage)

    print("args: ", args)
    np.random.seed(args.seed)
    time.sleep(3)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    else:
        shutil.rmtree(args.out_dir)
        os.mkdir(args.out_dir)

    # ============== get dataset ==============
    # use tfrecord
    print("============== get dataloader ==============")
    train_loader = TfrecordsLoader(args.file_list, args, "train", "xin")
    test_loader = TfrecordsLoader(args.test_list, args, "test", "xin")
    print("============ get tfrecord train data ===============")
    train_loader.create_tfrecord()
    num_train_file = train_loader.num_file
    train_loader.calMeanShape()
    mean_shape = train_loader.meanShape
    print("============ get tfrecord test data ===============")
    test_loader.create_tfrecord()
    num_test_file = test_loader.num_file

    # run below in epoch for large dataset
    test_dataset = test_loader.get_tfrecords(test_loader.records_list[0])
    batch_test_dataset = test_dataset.batch(args.batch_size)

    epoch_size = 1 + (num_train_file // args.batch_size)
    print('Test number of examples: {}'.format(num_test_file))

    # ================== create models ================
    print("=================== create models ===============")
    model = XinNingNetwork(args.num_labels, args.image_size, mean_shape, train_stage)
    # import pdb;pdb.set_trace()
    # get_model_summary(model, [args.image_size, args.image_size, 3])
    boundaries = [
        int(bound) * epoch_size for bound in args.lr_epoch.split(',')]
    lr_sc = [args.learning_rate * (0.1 ** b_i)
             for b_i in range(len(boundaries))]
    lr_sc.append(0.00000001)
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, lr_sc)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn(0))
    # ============ resotre pretrain =============
    print("================= resotre pretrain if exist =================")
    if args.pretrained_model:
        # pretrained_model = args.pretrained_model
        # root = tf.train.Checkpoint(optimizer=optimizer, model=model)
        # root.restore(tf.train.latest_checkpoint(pretrained_model))
        # print('Restore from model directory: {}'.format(pretrained_model))
        # import pdb;pdb.set_trace()
        model = tf.keras.models.load_model(os.path.join(args.pretrained_model, ("SavedModel")))

    # import pdb;pdb.set_trace()
    print("test start")
    test(batch_test_dataset, num_test_file, model, args, mean_shape, train_stage)


@tf.function
def test_step(model, image_batch):
    return model(image_batch, training=False)[0]

def test(batch_test_dataset, num_test_file, model, args, mean_shape, train_stage):
    loss_sum = 0
    _NRMSE = 0
    landmark_error = 0
    landmark_01_num = 0
    all_num = 0

    epoch_size = math.ceil(num_test_file * 1.0 / args.batch_size)
    print("num test file: ", num_test_file)
    print("test epoch size: ", epoch_size)
    for i, (image_batch, landmarks_batch, attribute_batch, euler_batch) in enumerate(batch_test_dataset):  # batch_num
        print("start epoch: ", i)
        # import pdb;pdb.set_trace()
        outputs = test_step(model, image_batch)
        # import pdb;pdb.set_trace()
        # landmarks_pre = outputs[0].numpy()
        landmarks_pre = (outputs[0] + mean_shape).numpy()
        if train_stage == 'stage2':
            landmarks_pre += outputs[1].numpy()

        for k in range(len(landmarks_pre)):
            all_num += 1
            # save label image
            # import pdb;pdb.set_trace()
            one_image = image_batch[k]
            one_landmark_pre = landmarks_pre[k]#  * args.image_size
            one_landmark = landmarks_batch[k].numpy()#  * args.image_size

            annotate_landmarks_pre = one_landmark_pre.reshape(-1, 2) * args.image_size
            img_tmp = one_image.numpy().copy()
            img_tmp = (img_tmp * 127.5) + 127.5
            for (x, y) in annotate_landmarks_pre.astype(np.int32):
                cv2.circle(img_tmp, (x, y), 1, (0, 255, 0), 1)
            print(os.path.join(args.out_dir, str(all_num)+".jpg"))
            cv2.imwrite(os.path.join(args.out_dir, str(all_num)+".jpg"), img_tmp)

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

            # calculate error
            error_all_points = 0
            # num points
            for count_point in range(len(one_landmark_pre) // 2):
                error_diff = one_landmark_pre[(count_point * 2):(count_point * 2 + 2)] - \
                    one_landmark[(count_point * 2):(count_point * 2 + 2)]
                error = np.sqrt(np.sum(error_diff * error_diff))
                error_all_points += error

            # loss
            # import pdb;pdb.set_trace()
            diff = one_landmark_pre - one_landmark
            loss = np.sum(diff * diff)
            loss_sum += loss

            # RMSE
            one_landmark_2 = one_landmark.reshape(-1, 2)
            one_landmark_pre_2 = one_landmark_pre.reshape(-1, 2)
            # diff_land = one_landmark_pre.reshape(-1, 2) - landmark_2
            # dis_land = [(np.linalg.norm(one_diff)*args.image_size) * for one_diff in diff_land]
            # _RMSE = np.mean(dis_land)
            #eye_diff_land = one_landmark_pre.reshape(-1, 2) - landmark.reshape(-1, 2)
            #eye_dis_land = [np.linalg.norm(one_diff) for one_diff in diff_land]
            # d_eyes = (np.linalg.norm(
            #     landmark_2[left_eye_edge] - landmark_2[right_eye_edge])*args.image_size)**2
            # import pdb;pdb.set_trace()
            # diff_2 = one_landmark_pre_2 - one_landmark_2
            # _RMSE = np.mean(np.sum(diff_2 * diff_2, 1))
            # diff_eyes = np.array([diff[left_eye_edge * 2], diff[left_eye_edge * 2 + 1], diff[right_eye_edge * 2], diff[right_eye_edge * 2 + 1]])
            # d_eyes = np.sum(diff_eyes * diff_eyes)
            # RMSE = _RMSE / d_eyes
            # print("RMSE: ", RMSE)
            # if np.isnan(RMSE) or np.isinf(RMSE):  # or RMSE > 50:
            #     import pdb
            #     pdb.set_trace()
            # _NRMSE += RMSE

            # time.sleep(3)
            interocular_distance = np.sqrt(
                np.sum(
                    pow((one_landmark[left_eye_edge*2:left_eye_edge*2+2] -
                            one_landmark[right_eye_edge*2:right_eye_edge*2+2]), 2)
                )
            )
            error_norm = error_all_points / \
                (interocular_distance * args.num_labels)
            # error_norm = error_all_points
            print("error_norm: ", error_norm)
            landmark_error += error_norm
            if error_norm >= 0.02:
                landmark_01_num += 1

    print("all_num: ", all_num)
    print("num_test_file: ", num_test_file)
    loss = loss_sum / (num_test_file * 1.0)
    print('Test epochs: {}\tLoss {:2.3f}'.format(epoch_size, loss))
    # NRMSE = _NRMSE / (num_test_file * 1.0)
    # print('Test NRMSE {:2.3f}'.format(NRMSE))

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

    parser.add_argument('--file_list', type=str,
                        default='data/train_data/list.txt')
    parser.add_argument('--test_list', type=str,
                        default='data/test_data/list.txt')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--max_epoch', type=int, default=10000)
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--num_labels', type=int, default=98)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default='models1/model_test')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_epoch', type=str,
                        default='20,30,40,60,100,160,180,200,500,990,1010')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--level', type=str, default='L5')
    parser.add_argument('--save_image_example', action='store_false')
    parser.add_argument('--debug', type=str, default='False')
    parser.add_argument('--depth_multi', type=float, default=1)
    parser.add_argument('--out_dir', type=str, default='sample_result')
    parser.add_argument('--num_quant', type=int, default=64)
    parser.add_argument('--tfrecords_dir', type=str, default='./tfrecords_xin')
    # parser.add_argument('--tfrecords_dir', type=str, default='./tfrecords_xin_gray')
    parser.add_argument('--is_augment', type=str2bool,
                        default=False, help='Whether to augment')

    return parser.parse_args(argv)


if __name__ == '__main__':
    print(sys.argv)
    main(parse_arguments(sys.argv[1:]))

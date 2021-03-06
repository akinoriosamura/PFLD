from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pfld import create_model
# from XinNing2020 import create_model
from generate_data import DataLoader


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import cv2
import shutil
import time
import argparse
import sys
import os


DEBUG = False

def get_mean_shape(args):
    file_base = os.path.basename(os.path.dirname(args.file_list))
    tfrecords_path = os.path.join(args.tfrecords_dir, file_base)
    mean_shape_txt = os.path.join(tfrecords_path, "mean_face_shape.txt")
    with open(mean_shape_txt, mode='r') as mf:
        mean_shape = mf.readline()
        mean_shape = mean_shape.split(" ")
        mean_shape = np.array([float(mf) for mf in mean_shape])

    return mean_shape

def main(args):
    test_type = "lip"
    print("============= test_type " + test_type + " =========")
    time.sleep(2)
    print("args: ", args)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    else:
        shutil.rmtree(args.out_dir)
        os.mkdir(args.out_dir)

    # get mean shape
    # mean_shape = get_mean_shape(args)

    loss_sum = 0
    NRMSE = 0
    landmark_error = 0
    landmark_01_num = 0
    lip_loss_sum = 0
    lip_upperloss_sum = 0

    with tf.Graph().as_default() as inf_g:
        image_batch = tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3),
                                     name='image_batch')
        landmark_batch = tf.placeholder(tf.float32, shape=(
            None, args.num_labels * 2), name='landmark_batch')

        phase_train_placeholder = tf.constant(False, name='phase_train')
        # landmarks_pre, _, _ = create_model(
        #     image_batch, landmark_batch, phase_train_placeholder, args)
        landmarks_pre, _, _ = create_model(
            image_batch, landmark_batch, phase_train_placeholder, args)

        save_params = tf.trainable_variables()
        saver = tf.train.Saver(save_params, max_to_keep=None)
        # quantize
        if args.num_quant < 64:
            print("=====================================")
            print("quantize by: ", args.num_quant)
            """
            tf.contrib.quantize.experimental_create_eval_graph(
                input_graph=inf_g,
                weight_bits=args.num_quant,
                activation_bits=args.num_quant,
                symmetric=False,
                quant_delay=None,
                scope=None
            )
            """
            # ref: https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/quantize
            tf.contrib.quantize.create_eval_graph(input_graph=inf_g)
        else:
            print("no quantize, so float: ", args.num_quant)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        inf_sess = tf.Session(
            graph=inf_g,
            config=tf.ConfigProto(
                gpu_options=gpu_options,
                allow_soft_placement=False,
                log_device_placement=False))
        inf_sess.run(tf.global_variables_initializer())
        inf_sess.run(tf.local_variables_initializer())

        with inf_sess.as_default():
            print('Model directory: {}'.format(args.pretrained_model))
            ckpt = tf.train.get_checkpoint_state(args.pretrained_model)
            print('ckpt: {}'.format(ckpt))
            model_path = ckpt.model_checkpoint_path
            assert (ckpt and model_path)
            epoch_start = int(
                model_path[model_path.find('model.ckpt-') + 11:]) + 1
            print('Checkpoint file: {}'.format(model_path))
            saver.restore(inf_sess, model_path)

            dataloader = DataLoader(args.test_list, args, "test")
            file_list, landmarks, attributes, euler_angles = dataloader.gen_data(
                args.test_list, args.num_labels)
            print(file_list)
            for file_id, file in enumerate(file_list):
                filename = os.path.split(file)[-1]
                image = cv2.imread(file)
                input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                input = cv2.resize(input, (args.image_size, args.image_size))
                input = input.astype(np.float32) / 256.0
                # input = (input.astype(np.float32) - 127.5) / 127.5
                input = np.expand_dims(input, 0)
                # print(input.shape)

                feed_dict = {
                    image_batch: input
                }
                st = time.time()
                pre_landmarks = inf_sess.run(
                    landmarks_pre, feed_dict=feed_dict)
                # print(pre_landmarks)
                print("elaps: ", time.time() - st)
                pre_landmark = pre_landmarks[0]
                # merge mean shape
                # pre_landmark += mean_shape
                # save labeled image
                h, w, _ = image.shape
                if DEBUG:
                    img = image.copy()
                    annotate_pre_landmark = pre_landmark.reshape(-1, 2) * [
                        h, w]
                    for land_id, (x, y) in enumerate(annotate_pre_landmark.astype(np.int32)):
                        cv2.circle(img, (x, y), 1, (0, 255, 0), 1)
                        cv2.imwrite("./show_labeled" +
                                    str(land_id) + ".jpg", img)
                    img = image.copy()
                    annotate_landmark = landmarks[file_id].reshape(-1, 2) * [
                        h, w]
                    for land_id, (x, y) in enumerate(annotate_landmark.astype(np.int32)):
                        cv2.circle(img, (x, y), 1, (0, 255, 0), 1)
                        cv2.imwrite("./show_test" + str(land_id) + ".jpg", img)
                    break
                    print(os.path.join(args.out_dir, filename))
                    cv2.imwrite(os.path.join(args.out_dir, filename), image)
                else:
                    annotate_pre_landmark = pre_landmark.reshape(-1, 2) * [
                        h, w]
                    for (x, y) in annotate_pre_landmark.astype(np.int32):
                        cv2.circle(image, (x, y), 1, (0, 255, 0), 1)
                    print(os.path.join(args.out_dir, filename))
                    cv2.imwrite(os.path.join(args.out_dir, filename), image)

                # cal loss
                landmark = landmarks[file_id]
                diff = pre_landmark - landmark
                loss = np.sum(diff * diff)
                loss_sum += loss

                # RMSE
                RMSE = np.sqrt(
                    np.mean(((landmark-pre_landmark)*args.image_size)**2))
                print("RMSE: ", RMSE)
                if np.isnan(RMSE) or np.isinf(RMSE):  # or RMSE > 50:
                    import pdb
                    pdb.set_trace()
                NRMSE += RMSE

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
                # print("eye: ", left_eye_edge)
                # print("eye; ", right_eye_edge)
                # print("labels: ", args.num_labels)
                # time.sleep(2)

                error_all_points = 0
                # num points
                for count_point in range(pre_landmark.shape[0] // 2):
                    error_diff = pre_landmark[(count_point * 2):(count_point * 2 + 2)] - \
                        landmark[(count_point * 2):(count_point * 2 + 2)]
                    error = np.sqrt(np.sum(error_diff * error_diff))
                    if (test_type == 'lip') and (48 <= count_point):
                        error *= 3
                        lip_loss_sum += error
                        if (48 <= count_point <= 54) or (60 <= count_point <= 64):
                            lip_upperloss_sum += error
                    error_all_points += error

                interocular_distance = np.sqrt(
                    np.sum(
                        pow((landmark[left_eye_edge*2:left_eye_edge*2+2] - landmark[right_eye_edge*2:right_eye_edge*2+2]), 2)
                        )
                )
                error_norm = error_all_points / (interocular_distance * args.num_labels)
                # error_norm = error_all_points
                print("error_norm: ", error_norm)
                landmark_error += error_norm
                if error_norm >= 0.02:
                    landmark_01_num += 1

            loss = loss_sum / (len(file_list) * 1.0)
            print('Test Loss {:2.3f}'.format(loss))
            #NRMSE = NRMSE / (len(file_list) * 1.0)
            #print('Test NRMSE {:2.3f}'.format(NRMSE))
            if (test_type == 'lip'):
                lip_loss = lip_loss_sum / (len(file_list) * 1.0)
                print('Test lip_loss {:2.3f}'.format(lip_loss))
                lip_upperloss = lip_upperloss_sum / (len(file_list) * 1.0)
                print('Test lip_upperloss {:2.3f}'.format(lip_upperloss))
            print('mean error and failure rate')
            landmark_error_norm = landmark_error / (len(file_list) * 1.0)
            error_str = 'mean error : {:2.3f}'.format(landmark_error_norm)
            failure_rate_norm = landmark_01_num / (len(file_list) * 1.0)
            failure_rate_str = 'failure rate: L1 {:2.3f}'.format(
                failure_rate_norm)
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

    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--file_list', type=str,
                        default='data/train_data/list.txt')
    parser.add_argument('--test_list', type=str,
                        default='data/test_data/list.txt')
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--num_labels', type=int, default=98)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_epoch', type=str, default='10,20,30,40,200,500')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--level', type=str, default='L5')
    parser.add_argument('--save_image_example', action='store_false')
    parser.add_argument('--depth_multi', type=float, default=1)
    parser.add_argument('--out_dir', type=str, default='sample_result')
    parser.add_argument('--tfrecords_dir', type=str, default='/data/tfrecords')
    parser.add_argument('--num_quant', type=int, default=64)
    parser.add_argument('--is_augment', type=str2bool,
                        default=False, help='Whether to augment')

    return parser.parse_args(argv)


if __name__ == '__main__':
    print(sys.argv)
    main(parse_arguments(sys.argv[1:]))

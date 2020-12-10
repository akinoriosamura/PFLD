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

def wing_loss(landmarks, labels, w=10.0, epsilon=2.0):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    print("=== wing loss === ")
    x = landmarks - labels
    c = w * (1.0 - math.log(1.0 + w/epsilon))
    absolute_x = tf.abs(x)
    #import pdb;pdb.set_trace()
    losses = tf.where(
        tf.greater(w, absolute_x),
        w * tf.math.log(1.0 + absolute_x/epsilon),
        absolute_x - c
    )

    return losses, [absolute_x, losses]

def main(args):
    train_stage = 'stage1'
    print("============= this phase is : ", train_stage)

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

    print('Total number of examples: {}'.format(num_train_file))
    print('Test number of examples: {}'.format(num_test_file))
    epoch_size = 1 + (num_train_file // args.batch_size)
    print('Number of batches per epoch: {}'.format(epoch_size))

    # ================== create models ================
    print("=================== create models ===============")
    model = XinNingNetwork(args.num_labels, args.image_size, mean_shape, train_stage)
    # import pdb;pdb.set_trace()
    if 'gray' in args.tfrecords_dir:
        _in_shape = [args.image_size, args.image_size, 1]
    else:
        _in_shape = [args.image_size, args.image_size, 3]
    print("_in_shape: ", _in_shape)
    get_model_summary(model, _in_shape)

    def loss_objects(outputs, targets, L2_losses):
        debug_list = []
        debug_list.append(outputs[0])
        # landmarks_pre = outputs[0]
        landmarks_pre = tf.add(tf.cast(mean_shape, dtype=tf.float32), outputs[0])
        if train_stage == 'stage2':
            debug_list.append(landmarks_pre)
            debug_list.append(outputs[1])
            landmarks_pre = tf.add(landmarks_pre, outputs[1])
            # landmarks_pre = outputs[1]
        debug_list.append(landmarks_pre)

        landmark_batch, euler_batch = targets[0], targets[1]
        # attributes_w_n = tf.to_float(attribute_batch[:, 1:6])
        # _num = attributes_w_n.shape[0]
        # mat_ratio = tf.reduce_mean(attributes_w_n, axis=0)
        # mat_ratio = tf.map_fn(lambda x: (tf.cond(x > 0, lambda: 1 / x, lambda: float(args.batch_size))), mat_ratio)
        # attributes_w_n = tf.convert_to_tensor(attributes_w_n * mat_ratio)
        # attributes_w_n = tf.reduce_sum(attributes_w_n, axis=1)

        L2_loss = tf.add_n(L2_losses)
        # _sum_k = tf.reduce_sum(tf.map_fn(lambda x: 1 - tf.cos(abs(x)), euler_angles_gt_batch - euler_angles_pre), axis=1)
        # default loss
        # print("=== default loss ===")
        # loss_sum = tf.reduce_sum(tf.square(landmark_batch - landmarks_pre), axis=1)
        _landmarks_pre = tf.reshape(landmarks_pre, [-1, args.num_labels, 2])
        _landmark_batch = tf.reshape(landmark_batch, [-1, args.num_labels, 2])
        # import pdb;pdb.set_trace()
        # wing loss
        print("=== wing loss ===")
        wing_losses, debug_l = wing_loss(_landmarks_pre, _landmark_batch)
        loss_sum = tf.reduce_sum(wing_losses, axis=[1, 2])
        # summarize
        loss_sum = tf.reduce_mean(loss_sum)  # * _sum_k)#  * attributes_w_n)
        loss_sum += L2_loss
        debug_list.append(landmark_batch - landmarks_pre)
        debug_list.append(tf.square(landmark_batch - landmarks_pre))
        debug_list.append(tf.reduce_sum(tf.square(landmark_batch - landmarks_pre), axis=1))
        debug_list.append(tf.reduce_mean(loss_sum))

        return loss_sum, L2_loss, debug_list

    boundaries = [
        int(bound) * epoch_size for bound in args.lr_epoch.split(',')]
    lr_sc = [args.learning_rate * (0.1 ** b_i)
             for b_i in range(len(boundaries))]
    lr_sc.append(0.00000001)
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, lr_sc)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn(0))

    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            outputs, heats = model(inputs, training=True)
            loss_value, L2_loss, debug_list = loss_objects(outputs, targets, model.losses)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return heats, loss_value, L2_loss, debug_list, outputs

    # ============ resotre pretrain =============
    print("================= resotre pretrain if exist =================")
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    # import pdb; pdb.set_trace()
    if args.pretrained_model:
        pretrained_model = args.pretrained_model
        manager = tf.train.CheckpointManager(ckpt, pretrained_model, max_to_keep=None)
        latest_ckpt_path = manager.latest_checkpoint
        ckpt.restore(latest_ckpt_path)
        print('Restore from model : {}'.format(latest_ckpt_path))
        if args.pretrained_model != args.model_dir:
            # in stage2 pretrain start
            epoch_start = 0
        else:
            epoch_start = int(latest_ckpt_path[latest_ckpt_path.find('ckpt-') + 5:]) + 1
    else:
        print("no pretrained exist")
        epoch_start = 0
    manager = tf.train.CheckpointManager(ckpt, args.model_dir, max_to_keep=None)

    # model.compile(
    #     optimizer=optimizer,
    #     # I used logits as output from the last layer, hence this
    #     loss=loss_objects,
    #     metrics=['mae']
    # )
    # import pdb; pdb.set_trace()
    all_step = epoch_start * epoch_size
    print("all_step: ", all_step)
    for epoch in range(epoch_start, args.max_epoch):
        batch_num = 0
        start_epoch = time.time()
        print("get dataset start")
        records_order = random.sample(
            train_loader.records_list, train_loader.num_records)
        assert len(records_order) == train_loader.num_records
        print("records order: ", records_order)
        for record_id, target_train_tfrecord_path in enumerate(records_order):
            if record_id != 0:
                print("delete dataset memory ")
                del train_dataset
                del batch_train_dataset
                del image_batch
                del landmarks_batch
                del euler_batch
                gc.collect()
            print("target_train_tfrecord_path : ", target_train_tfrecord_path)
            train_dataset = train_loader.get_tfrecords(
                target_train_tfrecord_path)
            print("get train_dataset")
            batch_train_dataset = train_dataset.batch(args.batch_size)
            print("get batch_train_dataset")
            for batch_i, (image_batch, landmarks_batch, _, euler_batch) in enumerate(batch_train_dataset):
                # start = time.time()
                # import pdb; pdb.set_trace()
                heats, losses, L2_loss, debug_list, outputs = train_step(
                    image_batch, [landmarks_batch, euler_batch])
                # print("trainable v num: ", len(model.trainable_variables))
                # outputs = model.fit(image_batch)
                # print("elapsed: ", time.time() - start)
                if ((batch_i + 1) % 10) == 0 or (batch_i + 1) == epoch_size:
                    Epoch = 'Epoch:[{:<4}][{:<4}/{:<4}][{:<4}/{:<4}]'.format(
                        epoch, record_id + 1, train_loader.num_records, batch_num, epoch_size)
                    # import pdb;pdb.set_trace()
                    Loss = 'Loss {:2.3f} L2 loss {:2.3f}'.format(
                        losses, L2_loss),
                    print('{}\t{}\t lr {:2.3}'.format(
                        Epoch, Loss, optimizer.learning_rate))
                    # save sample image
                    # import pdb;pdb.set_trace()
                    _sample_img = image_batch[0].numpy() * 127.5 + 127.5
                    _sampel_heatmap = 256 * np.concatenate([heats[1][0].numpy(), heats[1][0].numpy(), heats[1][0].numpy()], axis=2)
                    cv2.imwrite("./test_heatmap.jpg", (_sample_img + _sampel_heatmap))
                    _sample_stage1_land = debug_list[1][0].numpy().reshape(-1, 2) * 112
                    for (x, y) in _sample_stage1_land.astype(np.int32):
                        cv2.circle(_sample_img, (x, y), 1, (0, 255, 0), 1)
                    cv2.imwrite("./test_land.jpg", _sample_img)
                    if train_stage == 'stage2':
                        _sample_img_2 = (image_batch[0].numpy() * 127.5 + 127.5).copy()
                        _sample_stage2_land = debug_list[3][0].numpy().reshape(-1, 2) * 112
                        for (x, y) in _sample_stage2_land.astype(np.int32):
                            cv2.circle(_sample_img_2, (x, y), 1, (0, 255, 0), 1)
                        cv2.imwrite("./test_land2.jpg", _sample_img_2)
                    #import pdb;pdb.set_trace()

                batch_num += 1
                all_step += 1
                optimizer.learning_rate = learning_rate_fn(all_step)

        save_checkpoint_path = manager.save(checkpoint_number=epoch)
        print("save checkpoint: {}".format(save_checkpoint_path))
        savedmodel_path = os.path.join(model_dir, 'SavedModel/')
        tf.saved_model.save(model, savedmodel_path)
        print("save SavedModel: {}".format(savedmodel_path))
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
        tflite_model = converter.convert()
        with open(os.path.join(model_dir, "xinning.tflite"), 'wb') as f:
            f.write(tflite_model)
        print("save tflite: {}".format(os.path.join(model_dir, "xinning.tflite")))
        print("trainable v num: ", len(model.trainable_variables))
        print("elapsed epoch: ", time.time() - start_epoch)

        if epoch % 20 == 0 and epoch != 0 and epoch > 0:
            # import pdb;pdb.set_trace()
            print("test start")
            start = time.time()
            test(batch_test_dataset, num_test_file, model, args, mean_shape, train_stage)
            print("test time: {}" .format(time.time() - start))


@tf.function
def test_step(model, image_batch):
    return model(image_batch, training=False)[0]

def test(batch_test_dataset, num_test_file, model, args, mean_shape, train_stage):
    loss_sum = 0
    landmark_error = 0
    landmark_01_num = 0

    epoch_size = math.ceil(num_test_file * 1.0 / args.batch_size)
    print("num test file: ", num_test_file)
    print("test epoch size: ", epoch_size)
    for i, (image_batch, landmarks_batch, attribute_batch, euler_batch) in enumerate(batch_test_dataset):  # batch_num
        print("start epoch: ", i)
        outputs = test_step(model, image_batch)
        # import pdb;pdb.set_trace()
        # landmarks_pre = outputs[0].numpy()
        landmarks_pre = (outputs[0] + mean_shape).numpy()
        if train_stage == 'stage2':
            landmarks_pre += outputs[1].numpy()
            # landmarks_pre = outputs[1].numpy()
        diff = landmarks_pre - landmarks_batch
        loss = np.sum(diff * diff)
        loss_sum += loss

        for k in range(landmarks_pre.shape[0]):
            error_all_points = 0
            # num points
            for count_point in range(landmarks_pre.shape[1] // 2):
                error_diff = landmarks_pre[k][(count_point * 2):(count_point * 2 + 2)] - \
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
                # time.sleep(3)
                interocular_distance = np.sqrt(
                    np.sum(
                        pow((landmarks_batch[k][left_eye_edge*2:left_eye_edge*2+2] -
                             landmarks_batch[k][right_eye_edge*2:right_eye_edge*2+2]), 2)
                    )
                )
                error_norm = error_all_points / \
                    (interocular_distance * args.num_labels)
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

# @tf.function
def get_model_summary(model, in_shape):
    # 「仮のモデル」をFunctional APIで生成する独自関数
    def get_functional_model(_model, in_shape):
        x = Input(shape=(112, 112, 3), name='layer_in')
        temp_model = tf.keras.Model(
            inputs=[x],
            outputs=_model.call(x),  # ※サブクラス化したモデルの`call`メソッドを指定
            name='subclassing_model')  # 仮モデルにも名前付け
        # import pdb;pdb.set_trace()
        return temp_model

    # Functional APIの「仮のモデル」を取得
    #f_model = get_functional_model(model, in_shape)
    # モデルの内容を出力
    model.build(input_shape=(None,in_shape[0],in_shape[1],in_shape[2])) # , training=True) 
    model.summary()
    # モデルの構成図を表示
    # tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
    print("weights:", len(model.weights))
    print("trainable weights:", len(model.trainable_weights))
    print("===== save model summury ======")


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
    parser.add_argument('--max_epoch', type=int, default=81)
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--num_labels', type=int, default=98)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default='models1/model_test')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_epoch', type=str,
                        default='20,30,40,60,80,100,110,120,990,1010')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--level', type=str, default='L5')
    parser.add_argument('--save_image_example', action='store_false')
    parser.add_argument('--debug', type=str, default='False')
    parser.add_argument('--depth_multi', type=float, default=1)
    parser.add_argument('--num_quant', type=int, default=64)
    parser.add_argument('--tfrecords_dir', type=str, default='./tfrecords_xin')
    # parser.add_argument('--tfrecords_dir', type=str, default='./tfrecords_xin_gray')
    parser.add_argument('--is_augment', type=str2bool,
                        default=False, help='Whether to augment')

    return parser.parse_args(argv)


if __name__ == '__main__':
    print(sys.argv)
    main(parse_arguments(sys.argv[1:]))

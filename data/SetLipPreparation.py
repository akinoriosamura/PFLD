# -*- coding: utf-8 -*-
from euler_angles_utils import calculate_pitch_yaw_roll
import os
import numpy as np
import cv2
import shutil
import sys
import configparser

debug = False
# debug = True


def apply_rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1 - alpha) * center[0] - beta * center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta * center[0] + (1 - alpha) * center[1]

    landmark_ = np.asarray([(M[0, 0] * x + M[0, 1] * y + M[0, 2],
                             M[1, 0] * x + M[1, 1] * y + M[1, 2]) for (x, y) in landmark])
    return M, landmark_


class ImageDate():
    def __init__(self, line, imgDir, image_size, dataset):
        self.image_size = image_size
        line = line.strip().split()
        """
        label(175) = [136(68*2) points] + [4 bbox] + [6 attributes] + [28(14*2) pcn landmark] saveName
        """
        if len(line) != 175:
            import pdb
            pdb.set_trace()
        self.tracked_points = [17, 21, 22, 26,
                               36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
        self.list = line
        self.landmark = np.asarray(
            list(map(float, line[:136])), dtype=np.float32).reshape(-1, 2)
        self.landmark_lip = self.landmark[48:]
        # print("lip land num: ", len(self.landmark_lip))
        self.box = np.asarray(list(map(int, line[136:140])), dtype=np.int32)
        flag = list(map(int, line[140:146]))
        flag = list(map(bool, flag))
        self.pose = flag[0]
        self.expression = flag[1]
        self.illumination = flag[2]
        self.make_up = flag[3]
        self.occlusion = flag[4]
        self.blur = flag[5]
        self.pcn_landmark = np.asarray(
            list(map(float, line[146:174])), dtype=np.float32).reshape(-1, 2)
        self.pcn_landmark_lip = self.pcn_landmark[3:5]
        if imgDir == 'None':
            self.path = line[-1]
        else:
            self.path = os.path.join(imgDir, line[-1])
        self.img = None
        self.debug = True
        self.imgs = []
        self.landmarks = []
        self.landmark_lips = []
        self.boxes = []

    def show_labels(self):
        img = cv2.imread(self.path)
        # WFLW bbox: x_min_rect y_min_rect x_max_rect y_max_rect
        cv2.rectangle(img, (self.box[0], self.box[1]),
                      (self.box[2], self.box[3]), (255, 0, 0), 1, 1)
        for x, y in self.landmark:
            cv2.circle(img, (x, y), 3, (0, 0, 255))

        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_data(self, is_train, rotate, repeat, mirror=None):
        if (mirror is not None):
            with open(mirror, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                mirror_idx = lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))
        # ========画像を顔枠ら辺でcropし、輪郭点を調査========
        print("start")
        img = cv2.imread(self.path)
        # crop枠のサイズ
        len_lip = abs(self.pcn_landmark_lip[0]
                      [0] - self.pcn_landmark_lip[1][0])
        center = (self.pcn_landmark_lip[0] + (self.pcn_landmark_lip[1] -
                                              self.pcn_landmark_lip[0]) / 2).astype(np.int32)
        boxsize = int(len_lip * 1.7)
        # crop枠の左上を原点とした中心までの座標
        xy = center - boxsize // 2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        try:
            height, width, _ = img.shape
        except Exception as e:
            import pdb
            pdb.set_trace()
        # crop枠の左上 or 画像の左上縁
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        # crop枠の右下 or 画像の右下縁
        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # 唇周辺で切り抜き
        # 口を開いてる可能性もあるので、正方形
        imgT = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            # 画像をコピーし周りに境界を作成
            imgT = cv2.copyMakeBorder(
                imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            # 顔枠サイズが0なら
            # imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            print("no face bbox")
        else:
            if self.debug:
                # 表示して確認
                img_tmp = imgT.copy()
                for x, y in ((self.landmark_lip - xy) + 0.5).astype(np.int32):
                    cv2.circle(img_tmp, (x, y), 1, (255, 0, 0))
                for x, y in (self.pcn_landmark_lip - xy + 0.5).astype(np.int32):
                    cv2.circle(img_tmp, (x, y), 1, (0, 255, 2))
                cv2.imwrite("./sample_lip_resized.jpg", img_tmp)
                # import pdb;pdb.set_trace()
            if is_train:
                # 学習データに対してはリサイズ
                imgT = cv2.resize(imgT, (self.image_size, self.image_size))
            # クロップサイズに輪郭点ラベルを合わせる
            if self.debug:
                # 表示して確認
                img_tmp = imgT.copy()
                for x, y in (self.landmark_lip + 0.5).astype(np.int32):
                    cv2.circle(img_tmp, (x, y), 2, (255, 0, 0))
                for x, y in (self.pcn_landmark_lip + 0.5).astype(np.int32):
                    cv2.circle(img_tmp, (x, y), 2, (0, 255, 2))
                cv2.imwrite("./sample_lip_resized1.jpg", img_tmp)
                # import pdb;pdb.set_trace()
            landmark_lip = (self.landmark_lip - xy) / boxsize
            print("get imgT and label")
            if not (landmark_lip <= 1).all() or not (landmark_lip >= 0).all():
                img_tmp = imgT.copy()
                for x, y in (landmark_lip * boxsize + 0.5).astype(np.int32):
                    cv2.circle(img_tmp, (x, y), 2, (255, 0, 0))
                cv2.imwrite("./sample_lip_except.jpg", img_tmp)
                print("save  unuse image")
            else:
                assert (landmark_lip >= 0).all(), str(
                    landmark_lip) + str([dx, dy])
                assert (landmark_lip <= 1).all(), str(
                    landmark_lip) + str([dx, dy])
                self.landmark_lips.append(landmark_lip)
                self.landmarks.append(self.landmark)
                self.imgs.append(imgT)
                if self.debug:
                    # 表示して確認
                    img_tmp = imgT.copy()
                    for x, y in (landmark_lip * self.image_size + 0.5).astype(np.int32):
                        cv2.circle(img_tmp, (x, y), 2, (255, 0, 0))
                    for x, y in (self.pcn_landmark_lip + 0.5).astype(np.int32):
                        cv2.circle(img_tmp, (x, y), 2, (0, 255, 2))
                    cv2.imwrite("./sample_lip_resized3.jpg", img_tmp)
                    # import pdb;pdb.set_trace()
                print("pre rotate")
                if rotate == "rotate" and is_train:
                    # =========データ拡張=========
                    num_repeated = 0
                    while len(self.imgs) < repeat:
                        if num_repeated > 1000:
                            # 正解ラベルが1000回続けて出なければrotateなしに
                            # import pdb;pdb.set_trace()
                            repeat = 0
                            print("repear set 0")
                            continue
                        # print("num of rotate img for repeat: ", len(self.imgs))
                        angle = np.random.randint(-20, 20)
                        cx, cy = center
                        # cx = cx + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                        # cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                        M, landmark_lip = apply_rotate(
                            angle, (cx, cy), self.landmark_lip)

                        rotate_imgT = cv2.warpAffine(
                            img, M, (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1)))
                        wh = np.ptp(landmark_lip, axis=0).astype(np.int32) + 1
                        # size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                        size = boxsize
                        xy = np.asarray(
                            (cx - size // 2, cy - size // 2), dtype=np.int32)
                        landmark_lip = (landmark_lip - xy) / size
                        if self.debug:
                            # 表示して確認
                            img_tmp = rotate_imgT.copy()
                            for x, y in (landmark_lip * self.image_size + 0.5).astype(np.int32):
                                cv2.circle(img_tmp, (x, y), 1, (255, 0, 0))
                            cv2.imwrite("./sample_rotated.jpg", img_tmp)
                            # import pdb;pdb.set_trace()
                        if (landmark_lip < 0).any() or (landmark_lip > 1).any():
                            # print("wrong lip landmark so continue")
                            num_repeated += 1
                            continue

                        x1, y1 = xy
                        x2, y2 = xy + size
                        height, width, _ = rotate_imgT.shape
                        dx = max(0, -x1)
                        dy = max(0, -y1)
                        x1 = max(0, x1)
                        y1 = max(0, y1)

                        edx = max(0, x2 - width)
                        edy = max(0, y2 - height)
                        x2 = min(width, x2)
                        y2 = min(height, y2)

                        rotate_imgT = rotate_imgT[y1:y2, x1:x2]
                        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                            rotate_imgT = cv2.copyMakeBorder(
                                rotate_imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                        rotate_imgT = cv2.resize(
                            rotate_imgT, (self.image_size, self.image_size))

                        if mirror is not None and np.random.choice((True, False)):
                            landmark_lip[:, 0] = 1 - landmark_lip[:, 0]
                            landmark_lip = landmark_lip[mirror_idx]
                            rotate_imgT = cv2.flip(rotate_imgT, 1)

                        self.landmark_lips.append(landmark_lip)
                        self.landmarks.append(self.landmark)
                        self.imgs.append(rotate_imgT)
                        if self.debug:
                            # 表示して確認
                            img_tmp = rotate_imgT.copy()
                            for x, y in (landmark_lip * self.image_size + 0.5).astype(np.int32):
                                cv2.circle(img_tmp, (x, y), 1, (255, 0, 0))
                            cv2.imwrite("./sample_lip_resized4.jpg", img_tmp)
                            # import pdb;pdb.set_trace()

    def save_data(self, path, prefix):
        # attributeは特にいじらず保存
        attributes = [self.pose, self.expression, self.illumination,
                      self.make_up, self.occlusion, self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        for i, (img, landmark_lip) in enumerate(zip(self.imgs, self.landmark_lips)):
            save_path = os.path.join(path, prefix + '_' + str(i) + '.png')
            assert not os.path.exists(save_path), save_path
            # imgsにcrop画像を保存
            cv2.imwrite(save_path, img)

            # tracked pointsからpitch yaw rollを計算し保存
            landmark = self.landmarks[i]
            euler_angles_landmark = []
            for index in self.tracked_points:
                euler_angles_landmark.append(
                    [landmark[index][0] * img.shape[0], landmark[index][1] * img.shape[1]])
            euler_angles_landmark = np.asarray(
                euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(
                euler_angles_landmark[0], self.image_size, self.image_size)
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_lip_str = ' '.join(
                list(map(str, landmark_lip.reshape(-1).tolist())))

            label = '{} {} {} {}\n'.format(
                save_path, landmark_lip_str, attributes_str, euler_angles_str)
            labels.append(label)
        return labels


def get_dataset_list(imgDir, outDir, landmarkDir, is_train, rotate, image_size, dataset):
    with open(landmarkDir, 'r') as f:
        lines = f.readlines()
        labels = []
        save_img = os.path.join(outDir, 'imgs')
        os.makedirs(save_img, exist_ok=True)

        if debug:
            lines = lines[:100]
        print("get file num: ", len(lines))
        for i, line in enumerate(lines):
            Img = ImageDate(line, imgDir, image_size, dataset)
            img_name = Img.path
            Img.load_data(is_train, rotate, 10, Mirror_file)
            _, filename = os.path.split(img_name)
            filename, _ = os.path.splitext(filename)
            label_txt = Img.save_data(save_img, str(i) + '_' + filename)
            labels.append(label_txt)
            print("fin get label of: ", i + 1)
            print(filename)
            if ((i + 1) % 100) == 0:
                print('file: {}/{}'.format(i + 1, len(lines)))

    with open(os.path.join(outDir, 'list.txt'), 'w') as f:
        for label in labels:
            f.writelines(label)

    print("processed image num: ", len(labels))


if __name__ == '__main__':
    """
    ・68点の輪郭点を持つデータセットへのpreprocess
    ・WFLWと同じデータ・アノテーション構造に
     - images
        - .jpg
        - ...
     - anotations
        - test anotaion list
            landmarkはそのまま
            label = [136(68*2) points] + [4 bbox] + [6 attributes] + [28(14*2) pcn landmark] saveName
        - train annotaion list
    """

    # ex: python SetPreparation.py WFLW 98
    if len(sys.argv) == 3:
        dataset = sys.argv[1]
        rotate = sys.argv[2]
    else:
        print("please set arg(dataset_name rotate) ex: python SetLipPreparation.py pcnWFLW nonrotate")
        print("if you use pcn dataset, add nonrotate")
        exit()

    root_dir = os.path.dirname(os.path.realpath(__file__))

    config = configparser.ConfigParser()
    config.read('preparate_config.ini')

    section = dataset + "_lip"
    imageDirs = config.get(section, 'imageDirs')
    # 左右反転対象ラベル
    # なくてもいい
    Mirror_file = config.get(section, 'Mirror_file')
    if Mirror_file == "None":
        Mirror_file = None
    landmarkTrainDir = config.get(section, 'landmarkTrainDir')
    landmarkTestDir = config.get(section, 'landmarkTestDir')
    landmarkTestName = config.get(section, 'landmarkTestName')
    outTrainDir = config.get(section, 'outTrainDir')
    outTestDir = config.get(section, 'outTestDir')
    if rotate == "rotate":
        print("rotate")
        outTrainDir = "rotated_" + outTrainDir
        outTestDir = "rotated_" + outTestDir
    else:
        print("non rotate")
        outTrainDir = "non_rotated_" + outTrainDir
        outTestDir = "non_rotated_" + outTestDir
    image_size = int(config.get(section, 'ImageSize'))
    print(imageDirs)
    print(Mirror_file)
    print(landmarkTrainDir)
    print(landmarkTestDir)
    print(landmarkTestName)
    print(outTrainDir)
    print(outTestDir)
    print(image_size)

    landmarkDirs = [landmarkTestDir, landmarkTrainDir]

    outDirs = [outTestDir, outTrainDir]
    for landmarkDir, outDir in zip(landmarkDirs, outDirs):
        outDir = os.path.join(root_dir, outDir)
        print(outDir)
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
        os.mkdir(outDir)
        if landmarkTestName in landmarkDir:
            is_train = False
        else:
            is_train = True
        imgs = get_dataset_list(
            imageDirs, outDir, landmarkDir, is_train, rotate, image_size, dataset)
    print('end')

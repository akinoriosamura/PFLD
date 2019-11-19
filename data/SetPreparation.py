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
    def __init__(self, line, imgDir, num_labels, image_size, dataset):
        self.image_size = image_size
        line = line.strip().split()
        """
        num_labels = 98
        #0-195: landmark 坐标点  196-199: bbox 坐标点;
        #200: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
        #201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        #202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        #203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        #204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        #205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        #206: 图片名称
        num_labels = 68
        #0-135: landmark 坐标点  136-139: bbox 坐标点(x, y, w, h);
        #140: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
        #141: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        #142: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        #143: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        #144: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        #145: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        #146: image path
        """
        if num_labels == 68:
            if dataset == "WFLW":
                line = self.remove_unuse_land(line)
            if len(line) != 147:
                import pdb;pdb.set_trace()
            assert(len(line) == 147)
            self.tracked_points = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
            self.list = line
            self.landmark = np.asarray(list(map(float, line[:136])), dtype=np.float32).reshape(-1, 2)
            self.box = np.asarray(list(map(int, line[136:140])), dtype=np.int32)
            flag = list(map(int, line[140:146]))
            flag = list(map(bool, flag))
            self.pose = flag[0]
            self.expression = flag[1]
            self.illumination = flag[2]
            self.make_up = flag[3]
            self.occlusion = flag[4]
            self.blur = flag[5]
            self.path = os.path.join(imgDir, line[146])
            self.img = None
            self.num_labels = num_labels
            debug = False
            if debug:
                self.show_labels()
        elif num_labels == 98:
            assert(len(line) == 207)
            self.tracked_points = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
            self.list = line
            self.landmark = np.asarray(list(map(float, line[:196])), dtype=np.float32).reshape(-1, 2)
            self.box = np.asarray(list(map(int, line[196:200])), dtype=np.int32)
            flag = list(map(int, line[200:206]))
            flag = list(map(bool, flag))
            self.pose = flag[0]
            self.expression = flag[1]
            self.illumination = flag[2]
            self.make_up = flag[3]
            self.occlusion = flag[4]
            self.blur = flag[5]
            self.path = os.path.join(imgDir, line[206])
            self.img = None
            self.num_labels = num_labels
            debug = False
            if debug:
                self.show_labels()
        else:
            print("len landmark is not invalid")
            exit()
        self.imgs = []
        self.landmarks = []
        self.boxes = []

    def remove_unuse_land(self, line):
        del_ago = [2, 4, 6, 8, 9, 11, 12, 14, 18, 20, 21, 23, 24, 26, 28, 30]
        del_left_eye_blow = [38, 39, 40, 41]
        del_right_eye_blow = [47, 48, 49, 50]
        del_eye = [62, 66, 70, 74]
        del_eye_center = [96, 97]
        dels = del_ago + del_left_eye_blow + del_right_eye_blow + del_eye + del_eye_center
        # 削除する際にインデックスがずれないように降順に削除していく
        dels.sort(reverse=True)
        for del_id in dels:
            del_id_y = del_id * 2 + 1
            del_id_x = del_id * 2
            line.pop(del_id_y)
            line.pop(del_id_x)

        return line

    def show_labels(self):
        img = cv2.imread(self.path)
        # WFLW bbox: x_min_rect y_min_rect x_max_rect y_max_rect
        cv2.rectangle(img, (self.box[0], self.box[1]), (self.box[2], self.box[3]), (255, 0, 0), 1, 1)
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
        xy = np.min(self.landmark, axis=0).astype(np.int32)
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh / 2).astype(np.int32)
        img = cv2.imread(self.path)
        # debug

        # 顔枠のサイズ
        boxsize = int(np.max(wh) * 1.2)
        # 顔枠の左上を原点とした中心までの座標
        xy = center - boxsize // 2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        try:
            height, width, _ = img.shape
        except Exception as e:
            import pdb;pdb.set_trace()
        # 顔枠の左上 or 画像の左上縁
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        # 顔枠の右下 or 画像の右下縁
        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # 顔枠で切り抜き
        imgT = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            # 画像をコピーし周りに境界を作成
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            # 顔枠サイズが0なら
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # 表示して確認
            for x, y in (self.landmark + 0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()
        if is_train:
            # 学習データに対してはリサイズ
            imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        # クロップサイズに輪郭点ラベルを合わせる
        landmark = (self.landmark - xy) / boxsize
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.imgs.append(imgT)
        self.landmarks.append(landmark)

        if rotate=="rotate":
            # =========データ拡張=========
            while len(self.imgs) < repeat:
                angle = np.random.randint(-20, 20)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = apply_rotate(angle, (cx, cy), self.landmark)

                imgT = cv2.warpAffine(img, M, (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1)))
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size // 2), dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if mirror is not None and np.random.choice((True, False)):
                    landmark[:, 0] = 1 - landmark[:, 0]
                    landmark = landmark[mirror_idx]
                    imgT = cv2.flip(imgT, 1)

                self.imgs.append(imgT)
                self.landmarks.append(landmark)

    def save_data(self, path, prefix):
        # attributeは特にいじらず保存
        attributes = [self.pose, self.expression, self.illumination, self.make_up, self.occlusion, self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        for i, (img, landmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert landmark.shape == (self.num_labels, 2)
            save_path = os.path.join(path, prefix + '_' + str(i) + '.png')
            assert not os.path.exists(save_path), save_path
            # imgsにcrop画像を保存
            cv2.imwrite(save_path, img)

            # tracked pointsからpitch yaw rollを計算し保存
            euler_angles_landmark = []
            for index in self.tracked_points:
                euler_angles_landmark.append([landmark[index][0] * img.shape[0], landmark[index][1] * img.shape[1]])
            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0], self.image_size, self.image_size)
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_str = ' '.join(list(map(str, landmark.reshape(-1).tolist())))

            label = '{} {} {} {}\n'.format(save_path, landmark_str, attributes_str, euler_angles_str)
            labels.append(label)
        return labels


def get_dataset_list(imgDir, outDir, landmarkDir, is_train, rotate, num_labels, image_size, dataset):
    with open(landmarkDir, 'r') as f:
        lines = f.readlines()
        labels = []
        save_img = os.path.join(outDir, 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)

        if debug:
            lines = lines[:100]
        print("get file num: ", len(lines))
        for i, line in enumerate(lines):
            Img = ImageDate(line, imgDir, num_labels, image_size, dataset)
            img_name = Img.path
            Img.load_data(is_train, rotate, 10, Mirror_file)
            _, filename = os.path.split(img_name)
            filename, _ = os.path.splitext(filename)
            label_txt = Img.save_data(save_img, str(i) + '_' + filename)
            labels.append(label_txt)
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
            bboxは要検討
            attributeは適当に
            #0-135: landmark 坐标点  136-139: bbox 坐标点;
            #140: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
            #141: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
            #142: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
            #143: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
            #144: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
            #145: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
            #146: image path
        - train annotaion list
    """

    # ex: python SetPreparation.py WFLW 98
    if len(sys.argv) == 4:
        dataset = sys.argv[1]
        num_labels = sys.argv[2]
        rotate = sys.argv[3]
    else:
        print("please set arg(dataset_name num_labels rotate) ex: python SetPreparation.py pcnWFLW 68 nonrotate")
        print("if you use pcn dataset, add nonrotate")
        exit()

    root_dir = os.path.dirname(os.path.realpath(__file__))

    config = configparser.ConfigParser()
    config.read('preparate_config.ini')

    section = dataset + "_" + num_labels
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
    if rotate=="rotate":
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
        imgs = get_dataset_list(imageDirs, outDir, landmarkDir, is_train, rotate, int(num_labels), image_size, dataset)
    print('end')

# -*- coding: utf-8 -*-
from euler_angles_utils import calculate_pitch_yaw_roll
import os
import numpy as np
import cv2
import shutil
import sys
import configparser

# sys.path.append('../')
# from FaceKit.PCN.PyPCN import build_init_detector, get_PCN_result, draw_result, get_label_dict
# detector = build_init_detector()

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
    def __init__(self, line, num_labels, image_size, dataset, crop_base):
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
        if num_labels == 52:
            if len(line) != (num_labels * 2 + 11):
                import pdb;pdb.set_trace()
            assert(len(line) == (num_labels * 2 + 11))
            self.tracked_points = [9, 11, 12, 14, 20, 23, 26, 29, 17, 19, 32, 38, 41, 4]
            self.list = line
            self.landmark = np.asarray(list(map(float, line[:num_labels * 2])), dtype=np.float32).reshape(-1, 2)
            self.box = np.asarray(list(map(int, line[num_labels * 2:num_labels * 2 + 4])), dtype=np.int32)
            flag = list(map(int, line[num_labels * 2 + 4: num_labels * 2 + 10]))
            flag = list(map(bool, flag))
            self.pose = flag[0]
            self.expression = flag[1]
            self.illumination = flag[2]
            self.make_up = flag[3]
            self.occlusion = flag[4]
            self.blur = flag[5]
            self.path = line[num_labels * 2 + 10]
            self.img = None
            self.num_labels = num_labels
            self.crop_base = crop_base
            self.debug_num = 0
            #if debug:
            #    self.show_labels()
        elif num_labels == 68:
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
            self.path = line[146]
            self.img = None
            self.num_labels = num_labels
            self.crop_base = crop_base
            self.debug_num = 0
            #if debug:
            #    self.show_labels()
        elif num_labels == 20:
            if len(line) != 51:
                import pdb;pdb.set_trace()
            assert(len(line) == 51)
            # self.tracked_points = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
            self.list = line
            self.landmark = np.asarray(list(map(float, line[:40])), dtype=np.float32).reshape(-1, 2)
            self.box = np.asarray(list(map(int, line[40:44])), dtype=np.int32)
            flag = list(map(int, line[44:50]))
            flag = list(map(bool, flag))
            self.pose = flag[0]
            self.expression = flag[1]
            self.illumination = flag[2]
            self.make_up = flag[3]
            self.occlusion = flag[4]
            self.blur = flag[5]
            self.path = line[50]
            self.img = None
            self.num_labels = num_labels
            self.crop_base = crop_base
            self.debug_num = 0
            #if debug:
            #    self.show_labels()
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
        os.makedirs("sample_show_labels", exist_ok=True)
        img = cv2.imread(self.path)
        # WFLW bbox: x_min_rect y_min_rect x_max_rect y_max_rect
        cv2.rectangle(img, (self.box[0], self.box[1]), (self.box[2], self.box[3]), (0, 255, 0), 1, 1)
        for x, y in self.landmark:
            cv2.circle(img, (x, y), 3, (0, 255, 0))

        cv2.imwrite("sample_show_labels/sample_" + os.path.basename(self.path) + ".jpg", img)


    def get_new_pcn_bb(self, label):
        pcn_scale = 1.5
        new_box_dict = label["bbox"]
        pcn_x = new_box_dict["x"]
        pcn_y = new_box_dict["y"]
        pcn_w = new_box_dict["w"]

        new_bb_w = int(pcn_w * pcn_scale)
        scale_size = (new_bb_w - pcn_w) // 2
        # crop image
        new_bb_x1 = pcn_x - scale_size
        new_bb_y1 = pcn_y - scale_size
        new_bb_x2 = new_bb_x1 + new_bb_w
        new_bb_y2 = new_bb_y1 + new_bb_w

        return [new_bb_x1, new_bb_y1, new_bb_x2, new_bb_y2]

    def debug_label(self, f_name, img, bb, landmark):
        img_tmp = img.copy()
        cv2.rectangle(img_tmp, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1, 1)
        for x, y in ((landmark) + 0.5).astype(np.int32):
            cv2.circle(img_tmp, (x, y), 2, (0, 255, 0))
        cv2.imwrite(f_name, img_tmp)


    def load_data(self, is_train, rotate, repeat, mirror=None):
        if (mirror is not None):
            with open(mirror, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                mirror_idx = lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))
        # ========画像を顔枠ら辺でcropし、輪郭点を調査========
        img = cv2.imread(self.path)
        try:
            height, width, _ = img.shape
        except Exception as e:
            import pdb;pdb.set_trace()
        if crop_base == "landmark":
            xy = np.min(self.landmark, axis=0).astype(np.int32)
            zz = np.max(self.landmark, axis=0).astype(np.int32)
            wh = zz - xy + 1
            center = (xy + wh / 2).astype(np.int32)

            # 顔枠のサイズ
            boxsize = int(np.max(wh) * 1.2)
            # 顔枠の左上を原点とした中心までの座標
            xy = center - boxsize // 2
            x1, y1 = xy
            x2, y2 = xy + boxsize
            if debug:
                bb_tmp = [x1, y1, x2, y2]
                self.debug_label("./sample_pcn2.jpg", img, bb_tmp, self.landmark)

        elif crop_base == "bb":
            x1 = self.box[0]
            y1 = self.box[1]
            x2 = self.box[2]
            y2 = self.box[3]
            boxsize = x2 - x1
            xy = np.array([x1, y1])
            wh = np.array([boxsize, boxsize])
            center = (xy + wh / 2).astype(np.int32)
            if debug:
                bb_tmp = [x1, y1, x2, y2]
                self.debug_label("./sample_pcn2.jpg", img, bb_tmp, self.landmark)


        elif crop_base == "pcn":
            # 顔枠で切り抜き
            face_count, windows = get_PCN_result(img, detector)
            label_dicts, _ = get_label_dict(face_count, windows)
            # landmark[0]と一番近い左上bbをもつlabelを採用
            land_chin0x = np.min(self.landmark, axis=0).astype(np.int32)[0]
            label_dict = None
            pre_diff_chin0 = 10000
            for label in label_dicts:
                diff_chin0 = abs(land_chin0x - label["bbox"]["x"])
                if diff_chin0 <= pre_diff_chin0:
                    label_dict = label
                    pre_diff_chin0 = diff_chin0

            if debug:
                w_tmp = label_dict["bbox"]["w"]
                bb_tmp = [label_dict["bbox"]["x"], label_dict["bbox"]["y"], label_dict["bbox"]["x"]+w_tmp, label_dict["bbox"]["y"]+w_tmp]
                self.debug_label("./sample_pcn.jpg", img, bb_tmp, self.landmark)

            new_bb = self.get_new_pcn_bb(label_dict)

            x1 = new_bb[0]
            y1 = new_bb[1]
            x2 = new_bb[2]
            y2 = new_bb[3]
            boxsize = new_bb[2] - new_bb[0]
            xy = np.array([x1, y1])
            wh = np.array([boxsize, boxsize])
            center = (xy + wh / 2).astype(np.int32)
            if debug:
                bb_tmp = [x1, y1, x2, y2]
                self.debug_label("./sample_pcn2.jpg", img, bb_tmp, self.landmark)

        else:
            print("crop base error")
            exit()

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
                # 表示して確認

        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            # 顔枠サイズが0なら
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # 表示して確認
            for x, y in (self.landmark + 0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()

        # クロップサイズに輪郭点ラベルを合わせる
        landmark = (self.landmark - xy) / boxsize
        if debug:
            bb_tmp = [1, 1, x2-x1-1, y2-y1-1]
            land_tmp = landmark * boxsize
            self.debug_label("./sample_pcn3.jpg", imgT, bb_tmp, land_tmp)
        if is_train:
            # 学習データに対してはリサイズ
            imgT = cv2.resize(imgT, (self.image_size, self.image_size))

        try:
            pass
            # assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
            # assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        except:
            if debug:
                bb_tmp = [x1, y1, x2, y2]
                land_tmp = self.landmark - xy
                self.debug_label("./sample_pcn4.jpg", imgT, bb_tmp, land_tmp)
            import pdb; pdb.set_trace()
        self.imgs.append(imgT)
        self.landmarks.append(landmark)

        if rotate=="rotate" and is_train:
            # =========データ拡張=========
            repeat_num = 0
            while len(self.imgs) < repeat:
                repeat_num += 1
                if repeat_num > 1000:
                    break
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

                if debug:
                    # 表示して確認
                    self.debug_num += 1
                    os.makedirs("sample_labels", exist_ok=True)
                    img_tmp = imgT.copy()
                    for x, y in (landmark * self.image_size + 0.5).astype(np.int32):
                        cv2.circle(img_tmp, (x, y), 1, (255, 0, 0))
                    cv2.imwrite(os.path.join("sample_labels", "sample_" + str(self.debug_num) + ".jpg"), img_tmp)

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


def get_dataset_list(outDir, landmarkDir, is_train, rotate, num_labels, image_size, dataset, crop_base):
    with open(landmarkDir, 'r') as f:
        lines = f.readlines()
        labels = []
        save_img = os.path.join(outDir, 'imgs')
        os.makedirs(save_img, exist_ok=True)

        if debug:
            lines = lines[:100]
        print("get file num: ", len(lines))
        for i, line in enumerate(lines):
            if len(line.strip().split()) != num_labels * 2 + 11:
                print("error num of line in :")
                print(line)
                continue
            Img = ImageDate(line, num_labels, image_size, dataset, crop_base)
            img_name = Img.path
            if not os.path.exists(img_name):
                print("path is not exists: ")
                print(img_name)
                continue
            Img.load_data(is_train, rotate, 5, Mirror_file)
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
    print("DEBUG FLG: ", debug)
    if len(sys.argv) == 4:
        dataset = sys.argv[1]
        num_labels = sys.argv[2]
        rotate = sys.argv[3]
    else:
        print("please set arg(dataset_name num_labels rotate) ex: python SetPreparation.py pcnWFLW 68 nonrotate")
        print("if you use pcn dataset, add nonrotate")
        exit()

    config = configparser.ConfigParser()
    config.read('preparate_config.ini')

    section = dataset + "_" + num_labels
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
    crop_base = config.get(section, 'crop_base')
    print(Mirror_file)
    print(landmarkTrainDir)
    print(landmarkTestDir)
    print(landmarkTestName)
    print(outTrainDir)
    print(outTestDir)
    print(image_size)
    print(crop_base)
    # root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(landmarkTrainDir)

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
        imgs = get_dataset_list(outDir, landmarkDir, is_train, rotate, int(num_labels), image_size, dataset, crop_base)
    print('end')

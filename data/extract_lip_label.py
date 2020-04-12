# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import shutil
import sys


debug = False

class ImageDate():
    def __init__(self, line, img_dir):
        line = line.strip().split()
        """
        label(147) = [136(68*2) points] + [4 bbox] + [6 attributes] + saveName
        """
        if len(line) != 147:
            import pdb;pdb.set_trace()
        self.list = line
        self.landmark = np.asarray(list(map(float, line[:136])), dtype=np.float32).reshape(-1, 2)
        self.landmark_lip = self.landmark[48:]
        # print("lip land num: ", len(self.landmark_lip))
        self.box = np.asarray(list(map(int, line[136:140])), dtype=np.int32)
        self.flag = list(map(int, line[140:146]))
        if img_dir == "none":
            self.path = line[-1]
        else:
            self.path = os.path.join(img_dir, line[-1])
        self.img = None
        self.new_box = []

    def extract_lip(self):
        # ========画像を顔枠ら辺でcropし、輪郭点を調査========
        # crop枠のサイズ
        xy = np.min(self.landmark_lip, axis=0).astype(np.int32)
        zz = np.max(self.landmark_lip, axis=0).astype(np.int32)
        wh = zz - xy + 1
        center = (xy + wh / 2).astype(np.int32)
        boxsize = int(np.max(wh) * 1.2)
        # crop枠の左上を原点とした中心までの座標
        xy = center - boxsize // 2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        try:
            img = cv2.imread(self.path)
            height, width, _ = img.shape
        except Exception as e:
            import pdb;pdb.set_trace()
        # crop枠の左上 or 画像の左上縁
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)
        xy = (x1, y1)

        # crop枠の右下 or 画像の右下縁
        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        self.new_box = [x1, y1, x2, y2]

        if debug:
            # 表示して確認
            img_tmp = img.copy()
            cv2.rectangle(img, (self.new_box[0], self.new_box[1]), (self.new_box[2], self.new_box[3]), (255, 0, 0), 1, 1)
            for x, y in (self.landmark_lip + 0.5).astype(np.int32):
                cv2.circle(img_tmp, (x, y), 1, (255, 0, 0))
            cv2.imwrite("./sample_lip.jpg", img_tmp)
            # import pdb;pdb.set_trace()

    def save_data(self):
        # attributeは特にいじらず保存
        attributes = self.flag
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))

        bb_str = ' '.join(list(map(str, self.new_box)))
        self.landmark_lip = self.landmark_lip.astype(np.int32)
        landmark_lip_str = ' '.join(list(map(str, self.landmark_lip.reshape(-1).tolist())))

        label = '{} {} {} {}\n'.format(landmark_lip_str, bb_str, attributes_str, self.path)

        label_line = label.strip().split()
        if debug:
            print(len(label_line))
            print(label)
        if len(label_line) != 51:
            import pdb; pdb.set_trace()

        return label


if __name__ == '__main__':
    # change bb and landmark to around lip bb and lip landmark only of base txt label.
    # bedore : label(147) = [136(68*2) points] + [4 bbox] + [6 attributes] + saveName
    # after : label(51) = [40(20*2) points] + [4 bbox] + [6 attributes] + saveName
    if len(sys.argv) == 4:
        base_txt = sys.argv[1]
        # none or img dir path
        # if your label txt include img dir, set none
        img_dir = sys.argv[2]
        save_txt = sys.argv[3]
    else:
        print("please set arg(base_txt imgdir save_txt)")
        exit()

    with open(base_txt, 'r') as f:
        lines = f.readlines()
        labels = []

        if debug:
            lines = lines[:100]
        print("get file num: ", len(lines))
        for i, line in enumerate(lines):
            Img = ImageDate(line, img_dir)
            Img.extract_lip()
            label_txt = Img.save_data()
            labels.append(label_txt)
            if ((i + 1) % 100) == 0:
                print('file: {}/{}'.format(i + 1, len(lines)))

    with open(save_txt, 'w') as f:
        for label in labels:
            f.writelines(label)

    print("processed image num: ", len(labels))
    print('end')
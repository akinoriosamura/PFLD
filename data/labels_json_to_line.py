import os
import numpy as np
import cv2
import json
import sys

from sklearn.model_selection import train_test_split


def extract_annotations(json_dict):
    # crerate attributes and shape each label
    # save labels in lines
    annotations = []
    error_files = []
    correct_img_num = 0
    error_num = 0
    for img_file, label in json_dict.items():
        try:
            annotation = []
            # IMG_0555 2-1.jpg注意
            img_file = img_file.replace(" ", "_")
            # bb
            # {'top': '49', 'left': '49', 'width': '193', 'height': '194'}
            box = label["bb"]
            bbox = [int(box["top"]), int(box["left"]), int(box["width"]), int(box["height"])]
            bbox = list(map(str, bbox))
            for land_id in range(len(label["landmark"].values())):
            # for _, lands in label["landmark"].items():
                # , {'x': '93', 'y': '180'}
                lands = label["landmark"][str(land_id)]
                landmark = [int(lands['x']), int(lands['y'])]
                landmark = list(map(str, landmark))
                annotation.extend(landmark)
            # print("length landmark")
            # print(len(annotation))
            annotation.extend(bbox)
            # random attributes
            attributes = [0] * 6
            attributes = list(map(str, attributes))
            annotation.extend(attributes)
            annotation.extend([img_file])
            assert len(annotation) == 147

            annotations.append(annotation)
            correct_img_num += 1
        except Exception as e:
            error_num += 1
            print("例外args:", e.args)
            print(img_file)
            error_files.append(img_file)
            continue

    print("correct img num: ", correct_img_num)
    print("error img num: ", error_num)

    return annotations


if __name__ == '__main__':
    """
    Args:
        json: json path,
            {
                "7_180205_yoru_2000(1)(処理後).jpg": {
                    "bb": {
                        "left": 865.0,
                        "top": 126.0,
                        "width": 237.0,
                        "height": 310.0
                    },
                    "landmark": {
                        "0": {
                            "x": 868.0,
                            "y": 257.0
                        },
                        "1": {
                            "x": 868.0,
                            "y": 286.0
                        },
        img_dir: image dir

    growing dataはタブがファイル名にあるので変換の必要あり
    ex: find . -name "* *" | rename 's/ /_/g'
    ex: python labels_json_to_line.py /data/dataset/growing/traindata8979_20180601.json /data/dataset/growing/growing_20180601

    Save:
        txt: label lines text(train text : test text = 90 : 10)
            - landmarkはそのまま
            - bboxは要検討
            - attributeは適当に
            #0-135: landmark 坐标点  136-139: bbox 坐标点;
            #140: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
            #141: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
            #142: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
            #143: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
            #144: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
            #145: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
            #146: image path
    """
    # get labels
    DEBUG = True
    if len(sys.argv) == 3:
        json_f = sys.argv[1]
        img_dir = sys.argv[2]
    else:
        print("error: please write json_f path and img dir")
        exit()

    with open(json_f) as f:
        json_dict = json.load(f)
        annotations = extract_annotations(json_dict)

    if DEBUG:
        anno = annotations[0]
        landmarks = np.asarray(list(map(float, anno[:136])), dtype=np.float32).reshape(-1, 2)
        bbox = np.asarray((anno[136:140]), dtype=np.int32)

        def sort_box(_bbox):
            # WFLW bbox: x_min_rect y_min_rect x_max_rect y_max_rect
            # growing: top="49" left="49" width="193" height="194"
            _bbox = [_bbox[1], _bbox[0], _bbox[1] + _bbox[2], _bbox[0] + _bbox[3]]
            return _bbox
        # bbox = np.asarray(list(map(sort_box, bbox)))
        attribs = list(map(int, anno[140:146]))
        img_path = os.path.join(img_dir, anno[146])
        img = cv2.imread(img_path)
        # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1, 1)

        id = 1
        for x, y in landmarks:
            cv2.circle(img, (x, y), 3, (0, 255, 0))
            cv2.imwrite("./show_labeled" + str(id) + ".jpg", img)
            id += 1

    annotations = np.array(annotations)

    train_annos, test_annos = train_test_split(annotations, test_size=0.01)

    # save train text
    save_path = json_f[:-5] + "_train.txt"
    with open(save_path, mode='w') as f:
        for idx, anno in enumerate(train_annos):
            str_anno = " ".join(anno) + "\n"
            f.write(str_anno)

    # save test text
    save_path = json_f[:-5] + "_test.txt"
    with open(save_path, mode='w') as f:
        for idx, anno in enumerate(test_annos):
            str_anno = " ".join(anno) + "\n"
            f.write(str_anno)
    """
    # save all text
    save_path = json_f[:-5] + ".txt"
    with open(save_path, mode='w') as f:
        for idx, anno in enumerate(annotations):
            str_anno = " ".join(anno) + "\n"
            f.write(str_anno)
    print("finish save text labels")
    """
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
            bbox = [int(box["left"]), int(box["top"]), int(box["left"])+int(box["width"]), int(box["top"])+int(box["height"])]
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
    # get labels
    DEBUG = False
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
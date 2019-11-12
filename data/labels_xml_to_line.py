import os
import numpy as np
import cv2
import sys
import xml.etree.ElementTree as ET

from sklearn.model_selection import train_test_split


def extract_annotations(labels_xml):
    # crerate attributes and shape each label
    # save labels in lines
    annotations = []
    error_files = []
    correct_img_num = 0
    error_num = 0
    for img_xml in labels_xml[0].iter('image'):
        try:
            annotation = []
            # IMG_0555 2-1.jpg注意
            img_file = img_xml.attrib["file"].replace(" ", "_")
            # {'top': '49', 'left': '49', 'width': '193', 'height': '194'}
            box = img_xml[0].attrib
            bbox = [box["top"], box["left"], box["width"], box["height"]]
            bbox = list(map(str, bbox))
            landmarks = []
            for land_xml in img_xml[0]:
                # , {'name': '67', 'x': '93', 'y': '180'}
                landmark_ = land_xml.attrib
                landmark = [landmark_['x'], landmark_['y']]
                landmarks.append(landmark)
                landmark = list(map(str, landmark))
                annotation.extend(landmark)
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
        xml: xml path,
            <image file="image_071_1-0.jpg">
            <box top="49" left="49" width="193" height="194">
            <part name="00" x="59" y="109" />
            <part name="01" x="57" y="128" />
            <part name="02" x="58" y="148" />
            ...
        img_dir: image dir

    ex: python labels_xml_to_line.py /data/dataset/growing/traindata8979_20180601.xml /data/dataset/growing/growing_20180601

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
    DEBUG = False
    if len(sys.argv) == 3:
        xml = sys.argv[1]
        img_dir = sys.argv[2]
        tree = ET.parse(xml)
        labels_xml = tree.getroot()
    else:
        print("error: please write xml path and img dir")
        exit()

    annotations = extract_annotations(labels_xml)

    if DEBUG:
        annotations = annotations[:10]
        for anno in annotations:
            landmarks = np.asarray(list(map(float, anno[:136])), dtype=np.float32).reshape(-1, 2)
            bbox = np.asarray(list(map(int, anno[136:140])), dtype=np.int32)

            def sort_box(box):
                # WFLW bbox: x_min_rect y_min_rect x_max_rect y_max_rect
                # growing: top="49" left="49" width="193" height="194"
                box = [box[1], box[0], box[1] + box[2], box[0] + box[3]]
                return box
            bbox = np.asarray(list(map(sort_box, bbox)))
            import pdb; pdb.set_trace()
            attribs = list(map(int, anno[140:146]))
            img_path = os.path.join(img_dir, anno[146])
            img = cv2.imread(img_path)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1, 1)
            for x, y in landmarks:
                cv2.circle(img, (x, y), 3, (0, 0, 255))

            cv2.imshow("", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    annotations = np.array(annotations)
    train_annos, test_annos = train_test_split(annotations, test_size=0.1)

    # save train text
    save_path = xml[:-4] + "_train.txt"
    with open(save_path, mode='w') as f:
        for idx, anno in enumerate(train_annos):
            str_anno = " ".join(anno) + "\n"
            f.write(str_anno)

    # save test text
    save_path = xml[:-4] + "_test.txt"
    with open(save_path, mode='w') as f:
        for idx, anno in enumerate(test_annos):
            str_anno = " ".join(anno) + "\n"
            f.write(str_anno)

    print("finish save text labels")

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
    images_id = -1
    for id, label_xml in enumerate(labels_xml):
        if label_xml.tag == 'images':
            images_id = id
    image_num = 0
    for img_xml in labels_xml[images_id].iter('image'):
        img_file = img_xml.attrib["file"]
        # {'top': '49', 'left': '49', 'width': '193', 'height': '194'}
        for label_xml in img_xml:
            try:
                annotation = []
                box = label_xml.attrib
                bbox = [box["top"], box["left"], box["width"], box["height"]]
                bbox = list(map(str, bbox))
                landmarks = []
                for land_xml in label_xml:
                    # , {'name': '67', 'x': '93', 'y': '180'}
                    if land_xml.tag == 'label':
                        continue
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
                assert len(annotation) == 115

                annotations.append(annotation)
                image_num += 1
            except Exception as e:
                print("例外args:", e.args)
                print(img_file)
                error_files.append(img_file)
                continue
    print("total num: ", image_num)

    return annotations


if __name__ == '__main__':
    """
    Args:
        xml: xml path,
            if file has <?xml version='1.0' encoding='ISO-8859-1'?>
            remove encodeing above line
            <image file="image_071_1-0.jpg">
            <box top="49" left="49" width="193" height="194">
            <part name="00" x="59" y="109" />
            <part name="01" x="57" y="128" />
            <part name="02" x="58" y="148" />
            ...

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
    if len(sys.argv) == 2:
        xml = sys.argv[1]
        tree = ET.parse(xml)
        labels_xml = tree.getroot()
    else:
        print("error: please write xml path and img dir")
        exit()

    annotations = extract_annotations(labels_xml)

    annotations = np.array(annotations)
    train_annos, test_annos = train_test_split(annotations, test_size=0.01)

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

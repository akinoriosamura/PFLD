import os
import numpy as np
import cv2
import sys
import xml.etree.ElementTree as ET

from sklearn.model_selection import train_test_split


def del_68to52(lands):
    del_i = [2, 3, 6, 8, 10, 12, 15, 16, 19, 21, 24, 26, 29, 31, 33, 35]
    new_lands = []
    for id, land in enumerate(lands, 1):
        if id in del_i:
            continue
        new_lands.append(land)
    # import pdb; pdb.set_trace()

    return new_lands


def extract_annotations(labels_xml, label_num):
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
            annotation = []
            box = label_xml.attrib
            bbox = [box["left"], box["top"], str(
                int(box["left"])+int(box["width"])), str(int(box["top"])+int(box["height"]))]
            bbox = list(map(str, bbox))
            landmarks = []
            for land_xml in label_xml:
                # , {'name': '67', 'x': '93', 'y': '180'}
                if land_xml.tag == 'label':
                    continue
                landmark_ = land_xml.attrib
                landmark = [landmark_['x'], landmark_['y']]
                landmarks.append(landmark)
            landmarks = del_68to52(landmarks)
            for lands in landmarks:
                for land in lands:
                    annotation.append(str(land))
            annotation.extend(bbox)
            # random attributes
            attributes = [0] * 6
            attributes = list(map(str, attributes))
            annotation.extend(attributes)
            annotation.extend([img_file])
            if len(annotation) != 52*2+11:
                import pdb
                pdb.set_trace()
            try:
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
            ...
        img_dir: image dir

    Save:
        txt: label lines text(train text : test text = 90 : 10)
            - landmarkはそのまま
            - bbox: [x1, y1, x2, y2]
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
        img_dir = sys.argv[2]  # img_dir or dlib
        tree = ET.parse(xml)
        labels_xml = tree.getroot()
    else:
        print("error: please write xml path and img dir and datatype")
        exit()

    label_num = 68
    annotations = extract_annotations(labels_xml, label_num)
    import pdb
    pdb.set_trace()

    if DEBUG:
        annotations = annotations[:10]
        os.makedirs("checklabels_xml_line", exist_ok=True)
        for anno in annotations:
            landmarks = np.asarray(
                list(map(float, anno[:label_num*2])), dtype=np.float32).reshape(-1, 2)
            bbox = np.asarray(
                list(map(int, anno[label_num*2:label_num*2+4])), dtype=np.int32)

            """
            def sort_box(box):
                # WFLW bbox: x_min_rect y_min_rect x_max_rect y_max_rect
                box = [box[0], box[1], box[2], box[3]]
                return box
            bbox = np.asarray(list(map(sort_box, bbox)))
            """
            # import pdb; pdb.set_trace()
            attribs = list(map(int, anno[label_num*2+4:label_num*2+10]))
            if img_dir != "dlib":
                img_path = os.path.join(img_dir, anno[label_num*2+10])
            else:
                img_path = anno[label_num*2+10]
            img = cv2.imread(img_path)
            cv2.rectangle(img, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), (255, 0, 0), 1, 1)
            for x, y in landmarks:
                cv2.circle(img, (x, y), 3, (0, 0, 255))
            cv2.imwrite(os.path.join("checklabels_xml_line",
                                     "show_labeled" + os.path.basename(img_path) + ".jpg"), img)

        print("finish debug")
        exit()

    annotations = np.array(annotations)

    # save test text
    save_path = xml[:-4] + "_52.txt"
    with open(save_path, mode='w') as f:
        for idx, anno in enumerate(annotations):
            str_anno = " ".join(anno) + "\n"
            f.write(str_anno)
    print("total num: ", idx)

    print("finish save text labels")

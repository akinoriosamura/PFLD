import sys
import cv2
import numpy as np
import random
import os


if __name__ == "__main__":
    label_num = int(sys.argv[1])
    # pre processed txt label
    anno_path = sys.argv[2]  # labels txt or list.txt
    img_dir = sys.argv[3]  # img dir or dlib
    with open(anno_path, 'r') as f:
        lines = f.readlines()
    os.makedirs("checklabels", exist_ok=True)

    lines = random.sample(lines, k=20)
    # lines = lines[:5]
    if "list.txt" not in anno_path:
        print("label != list.txt")
        for line in lines:
            line = line.strip().split()
            landmarks = np.asarray(
                list(map(float, line[:label_num*2])), dtype=np.float32).reshape(-1, 2)
            # bbox = [x1, y1, x2, y2]
            bbox = np.asarray(
                list(map(float, line[label_num*2:label_num*2+4])), dtype=np.int32)
            # attribs = np.asarray(list(map(float, line[140:146])), dtype=np.int32)
            if img_dir == "dlib":
                img_name = line[-1]
            else:
                img_name = os.path.join(img_dir, line[-1])
            img = cv2.imread(img_name)
            print(img.shape)
            # bbox = [x1, y1, x2, y2]
            cv2.rectangle(img, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), (255, 0, 0), 1, 1)
            id = 1
            # import pdb; pdb.set_trace()
            for land_id, (x, y) in enumerate(landmarks):
                cv2.circle(img, (x, y), 1, (0, 255, 0))
                cv2.putText(img, str(
                    land_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), thickness=1)
                id += 1
            cv2.imwrite(os.path.join("checklabels", "show_labeled" +
                                     os.path.basename(img_name) + str(id) + ".jpg"), img)
    else:
        print("label == list.txt")
        for line in lines:
            line = line.strip().split()
            img_path = line[0]
            img = cv2.imread(img_path)
            print(img.shape)
            landmarks = np.asarray(
                list(map(float,  line[1:label_num*2+1])), dtype=np.float32)
            landmarks = landmarks.reshape(-1, 2)
            h, w, _ = img.shape
            # import pdb;pdb.set_trace()
            landmarks = np.asarray(landmarks * [h, w], np.int32)
            id = 1
            # import pdb; pdb.set_trace()
            for land_id, (x, y) in enumerate(landmarks):
                cv2.circle(img, (x, y), 1, (0, 255, 0))
                cv2.putText(img, str(
                    land_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), thickness=1)
                id += 1
            cv2.imwrite(os.path.join("checklabels", "show_labeled" +
                                     os.path.basename(img_path) + str(id) + ".jpg"), img)

    """
    label_num = int(sys.argv[1])
    # post processed txt label
    anno_path = sys.argv[2]
    with open(anno_path, 'r') as f:
        lines = f.readlines()

    line = lines[2].strip().split()
    img_path = line[0]
    # for lip 20 points
    landmarks = np.asarray(list(map(float,  line[1:label_num*2+1])), dtype=np.float32)
    # for 68points
    # landmarks = np.asarray(list(map(float, line[1:137])), dtype=np.float32)
    landmarks = landmarks.reshape(-1, 2)
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    landmarks = np.asarray(landmarks * [h, w], np.int32)
    id = 1
    for x, y in landmarks:
        cv2.circle(img, (x, y), 3, (0, 255, 0))
        cv2.imwrite("./show_labeled" + str(id) + ".jpg", img)
        id += 1

    print("finish")
    """

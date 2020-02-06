import sys
import cv2
import numpy as np
import os


if __name__ == "__main__":
    """
    anno_path = sys.argv[1] # labels txt 
    img_dir = sys.argv[2]
    with open(anno_path, 'r') as f:
        lines = f.readlines()
    os.makedirs("checklabels", exist_ok=True)

    # line = lines[1].strip().split()

    for line in lines:
        line = line.strip().split()
        landmarks = np.asarray(list(map(float, line[:136])), dtype=np.float32).reshape(-1, 2)
        bbox = np.asarray(list(map(float, line[136:140])), dtype=np.int32)
        attribs = np.asarray(list(map(float, line[140:146])), dtype=np.int32)
        img = cv2.imread(img_path)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1, 1)
        id = 1
        for x, y in landmarks:
            cv2.circle(img, (x, y), 3, (0, 255, 0))
            cv2.imwrite("./show_labeled" + str(id) + ".jpg", img)
            id += 1
        break
    """
    
    anno_path = sys.argv[1]
    with open(anno_path, 'r') as f:
        lines = f.readlines()

    line = lines[2].strip().split()
    img_path = line[0]
    landmarks = np.asarray(list(map(float, line[1:137])), dtype=np.float32)
    landmarks = landmarks.reshape(-1, 2)
    import pdb;pdb.set_trace()
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    landmarks = np.asarray(landmarks * [h, w], np.int32)
    id = 1
    for x, y in landmarks:
        cv2.circle(img, (x, y), 3, (0, 255, 0))
        cv2.imwrite("./show_labeled" + str(id) + ".jpg", img)
        id += 1

    print("finish")

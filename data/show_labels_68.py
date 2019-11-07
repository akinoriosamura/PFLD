import sys
import cv2
import numpy as np
import os


if __name__ == "__main__":
    anno_path = sys.argv[1]
    with open(anno_path, 'r') as f:
        lines = f.readlines()

    line = lines[0].strip().split()
    landmarks = np.asarray(list(map(float, line[:136])), dtype=np.float32).reshape(-1, 2)
    bbox = np.asarray(list(map(int, line[136:140])), dtype=np.int32)
    attribs = list(map(int, line[140:146]))
    img_path = os.path.join(img_dir, line[146])
    img = cv2.imread(img_path)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1, 1)
    for x, y in landmarks:
        cv2.circle(img, (x, y), 3, (0, 0, 255))

    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

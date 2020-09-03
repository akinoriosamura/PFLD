import numpy as np
import math
import cv2
import random
import time
import os

def _parse_data(fname):
    image = cv2.imread(fname)
    return image

with open('/data/prepro_dataset/rotated_train_pcn_aug_affine_300wCofwMenpoMultiXm2vtsWflwlGrow_68/list.txt', 'r') as f:
    lines = f.readlines()
    # import pdb;pdb.set_trace()
    exits_num = 0
    noexits_num = 0
    fnames = []
    for id, line in enumerate(lines):
        fname = line.strip().split()[0]
        if os.path.exists(fname):
            # print("exitst: ", str(id))
            exits_num += 1
            fnames.append(fname)
            # cv2.imread(fname)
        else:
            noexits_num += 1
            print("noexitst: ", fname)
            # import pdb;pdb.set_trace()
    print(exits_num)
    print(noexits_num)
    print(len(fnames))
    base_num = int(len(fnames) / 20)
    for j in range(20):
        # import pdb;pdb.set_trace()
        print("post index num: ", (j+1) * base_num)
        _fnames = fnames[j * base_num: (j+1) * base_num]
        print(len(_fnames))
        images = np.array(list(map(_parse_data, _fnames)))
        print(len(images))
    # import pdb;pdb.set_trace()
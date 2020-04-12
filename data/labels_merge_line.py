
import os
import sys
import json


if __name__ == "__main__":
    try:
        txt1_path = sys.argv[1]
        txt2_path = sys.argv[2]
    except:
        print("error: please write txt path and label_num and xml save path")

    with open(txt1_path, 'r') as f:
        lines_1 = f.readlines()
    with open(txt2_path, 'r') as f:
        lines_2 = f.readlines()

    import pdb;pdb.set_trace()

    lines = lines_1 + lines_2

    with open("sam.txt", mode='w') as f:
        for idx, anno in enumerate(lines):
            str_anno = " ".join(anno) + "\n"
            f.write(str_anno)

    print("num of txt1: ", len(lines_1))
    print("num of txt2: ", len(lines_2))
    print("total num: ", len(lines))
    print("total num: ", idx)

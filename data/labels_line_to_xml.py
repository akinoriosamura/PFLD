
import os
import sys
import numpy as np
import json


LABEL_NUM = 0


def emit_for_image(line):
    line = line.strip().split()
    landmarks = np.asarray(
        list(map(float, line[:LABEL_NUM*2])), dtype=np.float32).reshape(-1, 2)
    rect = np.asarray(
        list(map(float, line[LABEL_NUM*2:LABEL_NUM*2+4])), dtype=np.int32)
    # attribs = np.asarray(list(map(float, line[140:146])), dtype=np.int32)
    image_path = line[-1]

    landmark_xmls = []
    for i in range(len(landmarks)):
        if i < 10:
            index = "0" + str(i)
        else:
            index = str(i)
        x = int(round(landmarks[i][0]))
        y = int(round(landmarks[i][1]))

        landmark_xmls.append(
            '<part name="{0}" x="{1}" y="{2}" />'.format(index, x, y))
        face_xmls = '<box top="{0}" left="{1}" width="{2}" height="{3}">\n{4}\n</box>'.format(
            rect[1], rect[0], rect[2]-rect[0], rect[3] -
            rect[1], '\n'.join(landmark_xmls)
        )

    return '<image file="{0}">\n{1}\n</image>'.format(image_path, face_xmls)


if __name__ == "__main__":
    try:
        txt_path = sys.argv[1]
        LABEL_NUM = int(sys.argv[2])
        save_xml_path = sys.argv[3]
    except:
        print("error: please write txt path and label_num and xml save path")

    with open(txt_path, 'r') as f:
        lines = f.readlines()
    os.makedirs("checklabels", exist_ok=True)

    image_xmls = map(emit_for_image, lines)

    root_xml = '<dataset>\n<images>\n{0}\n</images>\n</dataset>'.format(
        '\n'.join(image_xmls))

    xml_file = open(save_xml_path, 'w')
    xml_file.write(root_xml)
    xml_file.close()

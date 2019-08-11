import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import argparse
from utils import *
from cascade_classifier import *


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 2.")
    parser.add_argument('data_dir', help='directory of images')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_dir = args.data_dir
    layer_file = './cascade_v2/cascade_layers.txt'
    cls_dir = './cascade_v2'
    kernels = np.load('./haar-(24,24).npy')
    # img_dir = './dataset/test_images_1000'
    files = os.listdir(img_dir)
    cascade = CascadeClassifier(layer_file, cls_dir)
    detector = Detector(cascade, kernels, 24, 24, shift=2)
    json_list = []
    save_path = './results2.json'
    for i, name in enumerate(files):
        print('detect: [{}], number {}'.format(name, i + 1))
        img = cv2.imread(os.path.join(img_dir, name))
        cur = time.time()
        bbox = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        print('using {}s, number of bboxes: {}'.format(time.time() - cur, len(bbox)))
        for bb in bbox:
            #     img = cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + round(bb[2] * bb[4]), bb[1] + round(bb[3] * bb[4])),
            #                          (0, 0, 255))
            json_list.append({"iname": name, "bbox": [bb[0], bb[1], round(bb[2] * bb[4]), round(bb[3] * bb[4])]})
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #  plt.imshow(img)
        # plt.show()
    with open(save_path, 'w') as f:
        json.dump(json_list, f)


if __name__ == '__main__':
    main()
    # print('start')
    # img = cv2.imread('./dataset/originalPics/2002/08/11/big/img_591.jpg')
    # img = cv2.imread('./dataset/originalPics/2002/07/25/big/img_1047.jpg')
    # draw = cv2.ellipse(img, (95, 45), (185, 121), 1.512147 * 180 / 3.14, 0, 360, (0, 0, 255))
    # print(1.265839 * 180 / 3.14)
    # theta = 1.265839
    # theta = 1.476417
    # a = 67
    # b = 44
    # print(max(a * np.cos(theta), b * np.sin(theta)))
    # draw = cv2.ellipse(draw, (267, 162), (123, 85), 0, 0, 360, (255, 0, 0))
    # p1, p2 = elli2rect(185.551587, 121.486235, 1.512147, 95.015995, 45.368737)
    # draw = cv2.rectangle(draw, p1, p2, (0, 255, 0))
    # p1, p2 = elli2rect(164.772023, 98.149591, -1.368127, 286.546014, 169.086526)
    # draw = cv2.rectangle(draw, p1, p2, (0, 255, 0))
    # cv2.imshow("", draw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imshow(draw)
    # plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from haar import *
from utils import *
import json


def is_overlap(box_object, box_target, wnd_size):
    """

    :param box_object: small
    :param box_target: large
    :param wnd_size: same window size (24)
    :return: overlap area / object area
    """
    x1 = box_object[0]
    y1 = box_object[1]
    scale1 = box_object[-1]
    x2 = box_target[0]
    y2 = box_target[1]
    scale2 = box_target[-1]

    S = wnd_size * wnd_size * scale1

    if x1 >= x2 and y1 >= y2:
        overlap = max(0, min(x2 + scale2 * wnd_size - x1, scale1 * wnd_size)) * max(0, min(y2 + scale2 * wnd_size - y1,
                                                                                           scale1 * wnd_size))
    elif x1 >= x2 and y1 < y2:
        overlap = max(0, min(x2 + scale2 * wnd_size - x1, scale1 * wnd_size)) * max(0, min(y1 + scale1 * wnd_size - y2,
                                                                                           scale2 * wnd_size))
    elif x1 < x2 and y1 >= y2:
        overlap = max(0, min(x1 + scale1 * wnd_size - x2, scale2 * wnd_size)) * max(0, min(y2 + scale2 * wnd_size - y1,
                                                                                           scale1 * wnd_size))
    elif x1 < x2 and y1 < y2:
        overlap = max(0, min(x1 + scale1 * wnd_size - x2, scale2 * wnd_size)) * max(0, min(y1 + scale1 * wnd_size - y2,
                                                                                           scale2 * wnd_size))
    else:
        return -1
    return overlap / S


class CascadeClassifier:
    def __init__(self, layer_file, cls_dir):
        self.layer_file = layer_file
        self.cls_dir = cls_dir
        self.layers = []
        with open(self.layer_file, 'r') as f:
            layer = f.readline()
            while layer:
                self.layers.append(layer.strip('\n'))
                layer = f.readline()

    def classify(self, wnd_integral, scale, kernels):
        for l, layer in enumerate(self.layers):
            ada_cls = np.load(os.path.join(self.cls_dir, layer))
            weak_pred = np.zeros((ada_cls.shape[0]))
            for i in range(ada_cls.shape[0]):
                index = int(ada_cls[i, 0])
                kernel_type = kernels[index, 0]
                kx = round(kernels[index, 1] * scale)
                ky = round(kernels[index, 2] * scale)
                pos_x = round(kernels[index, 3] * scale)
                pos_y = round(kernels[index, 4] * scale)

                white = get_sum_pixel(wnd_integral, kx, ky, pos_x, pos_y)
                if kernel_type == HaarType.TWO_V:
                    black = get_sum_pixel(wnd_integral, kx, ky // 2, pos_x, pos_y)
                elif kernel_type == HaarType.TWO_H:
                    black = get_sum_pixel(wnd_integral, kx // 2, ky, pos_x, pos_y)
                elif kernel_type == HaarType.THREE_V:
                    black = get_sum_pixel(wnd_integral, kx, ky // 3, pos_x, pos_y + ky // 3)
                elif kernel_type == HaarType.THREE_H:
                    black = get_sum_pixel(wnd_integral, kx // 3, ky, pos_x + kx // 3, pos_y)
                elif kernel_type == HaarType.FOUR_DIA:
                    black = get_sum_pixel(wnd_integral, kx // 2, ky // 2, pos_x + kx // 2, pos_y) \
                            + get_sum_pixel(wnd_integral, kx // 2, ky // 2, pos_x, pos_y + ky // 2)
                else:
                    raise ValueError('unknown kernel type')
                f = 2 * black - white
                threshold = ada_cls[i, 1]
                parity = ada_cls[i, 2]
                if (f - threshold) * parity > 0:
                    weak_pred[i] = 1
            if np.sum(weak_pred * ada_cls[:, 3]) - 0.5 * np.sum(ada_cls[:, 3]) > 0:
                continue
            else:
                return False
        return True


class Detector:
    def __init__(self, cascade_cls, kernels, wnd_x, wnd_y, shift=1):
        self.cascade_cls = cascade_cls
        self.kernels = kernels
        self.wnd_x = wnd_x
        self.wnd_y = wnd_y
        self.shift = shift

    def detect(self, image):
        max_scale = int(min(image.shape) / (2 * 24))
        print('max scale: {}'.format(max_scale))
        self.scales = [i for i in range(3, max_scale + 1, 2)]
        cur = time.time()
        img_integral = integral(image)[1:-1, 1:-1]
        print('compute integral using: {}s'.format(time.time() - cur))
        y, x = image.shape
        bounding_boxes = []
        for scale in self.scales:
            scale_boxes = []
            num_subset = []
            for move_y in range(0, y - self.wnd_y * scale, self.shift * scale):
                for move_x in range(0, x - self.wnd_x * scale, self.shift * scale):
                    wnd_integral = np.pad(
                        img_integral[move_y:move_y + self.wnd_y * scale, move_x:move_x + self.wnd_x * scale], 1,
                        mode='constant', constant_values=0).astype(np.int)
                    if self.cascade_cls.classify(wnd_integral, scale, self.kernels):
                        # print('detected, scale: {}'.format(scale))
                        # print(move_x, move_y, self.wnd_x, self.wnd_y, scale)
                        if len(scale_boxes) == 0:
                            # print('*' * 10, 'add new bbox', '*' * 10)
                            # scale_boxes.append([move_x, move_y, self.wnd_x, self.wnd_y, scale])
                            scale_boxes.append(
                                [move_x, move_y, move_x + self.wnd_x * scale, move_y + self.wnd_y * scale])
                            num_subset.append(1)
                        else:
                            add = True
                            # for i, box in enumerate(scale_boxes):
                            #     if abs(box[0] - move_x) <= 0.9 * self.wnd_x * box[4] and abs(
                            #             box[1] - move_y) <= 0.9 * self.wnd_y * box[4]:
                            #         scale_boxes[i][0] = int((box[0] * num_subset[i] + move_x) / (num_subset[i] + 1))
                            #         scale_boxes[i][1] = int((box[1] * num_subset[i] + move_y) / (num_subset[i] + 1))
                            #         num_subset[i] += 1
                            #         add = False
                            #         break
                            if add:
                                # print('*' * 10, 'add new bbox', '*' * 10)
                                # scale_boxes.append([move_x, move_y, self.wnd_x, self.wnd_y, scale])
                                scale_boxes.append(
                                    [move_x, move_y, move_x + self.wnd_x * scale, move_y + self.wnd_y * scale])
                                num_subset.append(1)
            tmp, _ = cv2.groupRectangles(scale_boxes, 2, 0.5)
            scale_boxes = []
            for t in tmp:
                scale_boxes.append(
                    [int(round(t[0])), int(round(t[1])), self.wnd_x, self.wnd_y,
                     ((t[2] - t[0]) / self.wnd_x + (t[3] - t[1]) / self.wnd_y) / 2])
            print('*' * 10, 'Scale: {}, bbox: {}'.format(scale, len(scale_boxes)), '*' * 10)
            if len(bounding_boxes) == 0:
                bounding_boxes = bounding_boxes + scale_boxes
            else:
                # merge variant scale bbox
                deleted_small = 0
                for i, new_box in enumerate(scale_boxes):
                    j = 0
                    while j < len(bounding_boxes):
                        box = bounding_boxes[j]
                        if is_overlap(box, new_box, 24) >= 0.3:
                            s1 = box[-1]
                            s2 = new_box[-1]
                            sum_scales = s1 + s2
                            scale_boxes[i][0] = round((s1 * box[0] + s2 * new_box[0]) / sum_scales)
                            scale_boxes[i][1] = round((s1 * box[1] + s2 * new_box[1]) / sum_scales)
                            scale_boxes[i][4] = (s1 * box[-1] + s2 * new_box[-1]) / sum_scales
                            new_box = scale_boxes[i]
                            bounding_boxes.pop(j)
                            deleted_small += 1
                        else:
                            j += 1
                print('*' * 10, 'deleted {} small bboxes'.format(deleted_small), '*' * 10)
                bounding_boxes = bounding_boxes + scale_boxes
        return bounding_boxes


if __name__ == '__main__':
    layer_file = './cascade_v2/cascade_layers.txt'
    cls_dir = './cascade_v2'
    kernels = np.load('./haar-(24,24).npy')
    # img_dir = './dataset/test_images_1000'
    img_dir = './dataset/originalPics/2002/07/25/big'
    files = os.listdir(img_dir)
    cascade = CascadeClassifier(layer_file, cls_dir)
    detector = Detector(cascade, kernels, 24, 24, shift=2)
    json_list = []
    save_path = './results.json'
    i = 0
    # for name in files:
    # if i >= 2:
    #    break
    # name = files[30]f
    for i, name in enumerate(['img_171.jpg', 'img_755.jpg', 'img_725.jpg', 'img_621.jpg']):
        img = cv2.imread(os.path.join(img_dir, name))
        cur = time.time()
        bbox = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        print('using {}s, number of bboxes: {}'.format(time.time() - cur, len(bbox)))
        for bb in bbox:
            img = cv2.rectangle(img, (int(bb[0]), int(bb[1])),
                                (int(bb[0] + round(bb[2] * bb[4])), int(bb[1] + round(bb[3] * bb[4]))), (0, 0, 255))
        # #     json_list.append({"iname": name, "bbox": [bb[0], bb[1], round(bb[2] * bb[4]), round(bb[3] * bb[4])]})
        cv2.imwrite('test{}.jpg'.format(i + 1), img)
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(img)
        # plt.title(name)
        # plt.show()
    # i += 1
    # with open(save_path, 'w') as f:
    #     json.dump(json_list, f)
    # cv2.groupRectangles()

import numpy as np
from utils import *
import os
import cv2


def generate_samples(img_dir, notate_files, save_dir, resize_shape=(24, 24), n_num=2):
    positive = []
    negative = []
    for notate_file in notate_files:
        with open(notate_file) as f:
            line = f.readline()
            img_path = ''
            face_num = 0
            while line:
                line = line.strip('\n')
                if len(line) < 5:
                    face_num = int(line)
                else:
                    if face_num == 0:
                        img_path = line
                        img = cv2.imread(os.path.join(img_dir, img_path + '.jpg'), 0)
                        n1 = img[:resize_shape[0] * 3, :resize_shape[1] * 3]
                        n2 = img[-resize_shape[0] * 3:, -resize_shape[1] * 3:]
                        n1 = cv2.resize(n1, resize_shape)
                        n2 = cv2.resize(n2, resize_shape)
                        negative.append(n1)
                        negative.append(n2)
                    else:
                        params = line.split(' ')
                        a = float(params[0])
                        b = float(params[1])
                        theta = float(params[2])
                        x = float(params[3])
                        y = float(params[4])
                        p1, p2 = elli2rect(a, b, theta, x, y)
                        # print(p1, p2)
                        p1[0] = max(0, p1[0])
                        p1[1] = max(0, p1[1])
                        p2[0] = min(p2[0], img.shape[1])
                        p2[1] = min(p2[1], img.shape[0])
                        face = img[p1[1]:p2[1] + 1, p1[0]:p2[0] + 1]
                        # print(face.shape)
                        face = cv2.resize(face, resize_shape)
                        positive.append(face)
                        face_num -= 1
                line = f.readline()
                print(os.path.join(img_dir, img_path + '.jpg'), face_num, len(positive), len(negative))
        print('*'*10, 'finished ', notate_file, '*'*10)
    np.save(os.path.join(save_dir, 'positive.npy'), np.asarray(positive))
    np.save(os.path.join(save_dir, 'negative.npy'), np.asarray(negative))


if __name__ == '__main__':
    # img_dir = './dataset/originalPics'
    # notate_files = ['./dataset/FDDB-folds/FDDB-fold-{:02d}-ellipseList.txt'.format(i) for i in range(1, 11)]
    # save_dir = './dataset'
    # generate_samples(img_dir, notate_files, save_dir)
    p = np.load('./dataset/positive.npy')
    n = np.load('./dataset/negative.npy')
    plt.figure()
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.imshow(n[i], cmap='gray')
    plt.show()
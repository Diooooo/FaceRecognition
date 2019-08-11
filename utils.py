import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import argparse


def integral(img):
    if len(img.shape) == 2:
        m, n = img.shape
    else:
        raise ValueError('image must be a single gray scale image')
    integral = np.pad(img, 1, mode='constant', constant_values=0).astype(np.int)
    # for i in range(1, m):
    #     integral[i, 0] = integral[i, 0] + integral[i - 1, 0]
    # for i in range(1, n):
    #     integral[0, i] = integral[0, i] + integral[0, i - 1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            integral[i, j] = integral[i, j] + integral[i, j - 1] + integral[i - 1, j] - integral[i - 1, j - 1]
    return integral


def get_sum_pixel(integral_img, kx, ky, x, y):
    if x + kx > integral_img.shape[1] - 2 or y + ky > integral_img.shape[0] - 2:
        raise ValueError('position out of range')
    return integral_img[y, x] + integral_img[y + round(ky), x + round(kx)] - integral_img[y + round(ky), x] - \
           integral_img[y, x + round(kx)]


def elli2rect(a, b, theta, x, y):
    x = int(round(x))
    y = int(round(y))

    theta = abs(theta)
    x_max = int(max(a * np.cos(theta), b * np.sin(theta)))
    y_max = int(max(a * np.sin(theta), b * np.cos(theta)))

    return [x - x_max, y - y_max], [x + x_max, y + y_max]

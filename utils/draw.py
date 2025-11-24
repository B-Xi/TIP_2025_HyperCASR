import os

import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt


def getColors_UP():
    return np.array([[0, 0, 255], [0, 255, 0], [0, 172, 254], [101, 193, 60],
                    [255, 0, 255], [147, 67, 46], [164, 75, 155], [60, 91, 112],
                    [255, 255, 0]])

def getColors_IP():
    return np.array([[0, 0, 255], [255, 100, 0], [164, 75, 155], [60, 91, 112],
                     [255, 255, 125], [255, 0, 255], [100, 0, 255], [0, 255, 0]])

def getColors_HU():
    return np.array([[147, 67, 46], [0, 0, 255], [255, 100, 0], [0, 255, 123],
                     [164, 75, 155], [118, 254, 172], [60, 91, 112], [255, 255, 0],
                     [255, 255, 125], [255, 0, 255], [100, 0, 255]])


def getClassificationMap(label: torch.tensor or np.ndarray, info, unknown=[]):
    if info['path'] == 'PaviaU':
        colors = getColors_UP()
    elif info['path'] == 'Indian_pines':
        colors = getColors_IP()
    elif info['path'] == 'Houston':
        colors = getColors_HU()
    image = np.zeros((*label.shape, 3), dtype='uint8')
    for cls in range(1, label.max() + 1):
        image[np.where(label == cls)] = colors[cls - 1]

    for cls in unknown:
        image[np.where(label == cls)] = [255, 255, 255]

    return image


def clearBackground(info, image):
    from scipy.io import loadmat
    dataset_path = os.path.join('/mnt/HDD/data/zwj/model_1/my_model/datasets', info["path"])
    gt = loadmat(os.path.join(dataset_path, info["gt_file_name"]))[info["gt_mat_name"]].astype(np.int64)

    image[np.where(gt == 0)] = [0, 0, 0]

    return image


def parsePredictionLabel(label: List[torch.tensor], H):
    label = np.concatenate(label) + 1
    return label.reshape(H, -1)


def drawPredictionMap(label: List[torch.tensor], info, draw_background=True):
    label = parsePredictionLabel(label, info['image_width'])
    image = getClassificationMap(label, info, unknown=[label.max()])

    if draw_background is False:
        image = clearBackground(info, image)
    saveImage(image, info['path'], '/mnt/HDD/data/zwj/model_1/my_model')


def saveImage(image, name, path='/mnt/HDD/data/zwj/model_1/my_model'):
    if not os.path.exists(path):
        os.makedirs(path)

    plt.imsave(f'{os.path.join(path, name)}.png', image)

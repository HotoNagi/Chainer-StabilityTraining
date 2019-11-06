#! /usr/bin/env python
# -*- conding:utf-8 -*-
import os
import sys
import chainer
import numpy as np

import cv2
import math
import random

from models import archs

from functools import partial
from chainercv import transforms


def get_dataset(train_data, test_data, root, datasets, use_mean=False):

    mean_path = root + '/mean.npy'
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        mean = compute_mean(datasets, root)
        np.save(mean_path, mean)
    print('use mean flag is ', use_mean)
    if not use_mean:
        print('not using mean')

    train = chainer.datasets.TransformDataset(
        train_data, partial(_transform2,
                            mean=mean, train=True, mean_flag=use_mean))
    test = chainer.datasets.TransformDataset(
        test_data, partial(_transform2,
                           mean=mean, train=False, mean_flag=use_mean))

    return train, test, mean


# 画像平均計算
def compute_mean(datasets, root, size=(224, 224)):

    print('画像平均計算...')
    sum_image = 0
    N = len(datasets)
    for i, (image, _) in enumerate(datasets):
        # imgのリサイズ
        image = image.transpose(1, 2, 0)
        image = resize(image, size)
        sum_image += image
        sys.stderr.write('{} / {}\r'.format(i+1, N))
        sys.stderr.flush()
    sys.stderr.write('\n')
    mean = sum_image / N

    return mean

# 前処理
def _transform2(data, mean, train=True, mean_flag=False):
    
    img, label = data
    img = img.copy()

    size316 = (316, 316)
    size = (224, 224)

    img_o = transforms.scale(img, 316)
    img_o = transforms.center_crop(img_o, size316)

    # 学習のときだけ実行
    if train:
        img_o = transforms.random_flip(img_o, y_random=True)
        img_o = transforms.random_rotate(img_o)
        # img = random_erase(img)

    img_o = transforms.resize(img_o, size)
    # 画像から平均を引く
    if mean_flag:
        img_o -= mean
    img_o *= (1.0 / 255.0)

    r = random.randint(316, 1500)
    img_st = transforms.scale(img, r)
    img_st = transforms.center_crop(img_st, (224, 224))
    # 画像から平均を引く
    if mean_flag:
        img_st -= mean
    img_st *= (1.0/255.0)

    return img_o, label, img_st


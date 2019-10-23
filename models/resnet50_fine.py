#! /usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from chainer.functions.array.reshape import reshape
from chainer import reporter
from chainercv import transforms

import random


# ResNet50(Fine-tuning)
def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = reshape(h, (n, channel))
    return h


class ResNet50_Fine(chainer.Chain):
    def __init__(self, output=8):
        super(ResNet50_Fine, self).__init__()

        with self.init_scope():
            self.base = L.ResNet50Layers()
            self.fc = L.Linear(None, output)

    def __call__(self, x):
        h = self.base(x, layers=['res5'])['res5']
        self.cam = h
        h = _global_average_pooling_2d(h)
        h = self.fc(h)
        
        x_z = Center_Crop(x)
        h_z = self.base(x_z, layers=['res5'])['res5']
        h_z = _global_average_pooling_2d(h_z)
        h_z = self.fc(h_z)

        return h, h_z

class StabilityClassifer(L.Classifier):
    def __init__(self, predictor,
                 lossfun=F.softmax_cross_entropy,
                 accfun=F.accuracy,):
        super(StabilityClassifer, self).__init__(predictor=predictor, lossfun=lossfun, accfun=accfun)

    def forward(self, x, t):

        if not chainer.config.train:
            self.y ,self.y_z = self.predictor(x)
            self.loss = self.lossfun(self.y, t)
        else:
            self.y, self.y_z= self.predictor(x)
            l1 = self.lossfun(self.y, t)
            self.l2 = compute_KLd(self.y, self.y_z)
            self.loss = l1 + (0.5 * self.l2)

        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


def compute_KLd(p, q):
        assert p.shape[0] == q.shape[0]
        p = F.softmax(p, axis=1)
        q = F.softmax(q, axis=1)
        return F.sum(p * (F.log(p + 1e-16) - F.log(q + 1e-16)))


def Center_Crop(x):
    for i in range(x.shape[0]):
        h = x.shape[2]
        r = random.randint(h//2, h)
        a = transforms.center_crop(x[i], (224, 224))
        x[i] = transforms.resize(a, (h, h))
    return x
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer import dataset
from chainer.training import extensions
from utils.get_dataset import compute_mean
from utils.get_dataset import _transform2

from chainercv.datasets import DirectoryParsingLabelDataset
from chainer.datasets import  LabeledImageDataset
from functools import partial

from models import archs
from utils.test_args import parser
from utils.confusion_matrix_cocoa import confusion_matrix_cocoa


def main():

    args = parser()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    root = args.dataset
    dataset = DirectoryParsingLabelDataset(root)
    mean_path = root + '/mean.npy'
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        mean = compute_mean(datasets, root)
        np.save(mean_path, mean)
    use_mean = args.use_mean
    print('use mean flag is ', use_mean)
    if not use_mean:
        print('not using mean')

    X = np.array([image_paths for image_paths in dataset.img_paths])
    y = np.array([label for label in dataset.labels])

    test_data = LabeledImageDataset([(x,y) for x,y in zip(X, y)])
    test = chainer.datasets.TransformDataset(
        test_data, partial(_transform2,
                           mean=mean, train=False, mean_flag=args.use_mean))
    #test = chainer.datasets.TransformDataset(test_data, _validation_transform)
    #test_model = L.Classifier(VGG16()).to_gpu()
    class_num = len(set(dataset.labels))
    model = L.Classifier(archs[args.arch](output=class_num)).to_gpu()
    
    serializers.load_npz(args.load_npz, model)

    dnames = glob.glob('{}/*'.format(root))
    labels_list = []
    for d in dnames:
        p_dir = Path(d)
        labels_list.append(p_dir.name)
    if 'mean.npy' in labels_list:
        labels_list.remove('mean.npy')
    confusion_matrix_cocoa(test, args.gpu, class_num,
                           model, save_dir, 1, labels_list)

if __name__ == '__main__':
    main()
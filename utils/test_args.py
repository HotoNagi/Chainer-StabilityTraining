#! /usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from models import archs

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        default='../bad',
                        help="Choise input dataset by full path")

    parser.add_argument('--gpu', '-g',
                        type=int, default=-0,
                        help="Number of using gpu_id")

    parser.add_argument('--arch', '-a',
                        choices=archs.keys(), default='resnet50_fine',
                        help="Name of using training model")

    parser.add_argument('--load_npz', '-l',
                        default='',
                        help="Name of using npz file")

    parser.add_argument('--save_dir', '-s',
                        default='test_result',
                        help="Save direction")
    
    parser.add_argument('--use_mean', '-m',
                        default=False,
                        help="Mean use")

    args = parser.parse_args()

    return args
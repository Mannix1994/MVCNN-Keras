# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

SEED = 1213

NUM_CLASSES = 250

NUM_VIEWS = 12

IMAGE_SHAPE = (227, 227, 3)

VIEWS_IMAGE_SHAPE = (NUM_VIEWS, 227, 227, 3)

TRAIN_LIST = './data/view/train_lists.txt'

VAL_LIST = './data/view/val_lists.txt'


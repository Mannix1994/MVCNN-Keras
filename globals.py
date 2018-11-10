# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

SEED = 1213

NUM_CLASSES = 40

NUM_VIEWS = 12

IMAGE_SHAPE = (227, 227, 3)

VIEWS_IMAGE_SHAPE = (NUM_VIEWS, 227, 227, 3)

IMAGE_DEPTH = 255

TRAIN_LIST = './data/train_lists.txt'

VAL_LIST = './data/val_lists.txt'

TRAIN_EPOCH_NUM = 10

TRAIN_BATCH_SIZE = 32

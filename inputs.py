# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import globals as _g
import model
import cv2


def prepare_dataset(name=''):
    # read image list files name and labels
    lists_and_labels = np.loadtxt(name, dtype=str).tolist()
    # shuffle dataset
    random.shuffle(lists_and_labels)
    # split lists an labels
    list_files, labels = zip(*[(l[0], int(l[1])) for l in lists_and_labels])

    # one_shot encoding labels
    one_shot_labels = keras.utils.to_categorical(labels, _g.NUM_CLASSES)
    one_shot_labels.astype(dtype=np.int32)

    def generator():
        for list_file, label in zip(list_files, one_shot_labels):
            image_list = np.loadtxt(list_file, dtype=str, skiprows=2)
            assert len(image_list) == _g.NUM_VIEWS
            images = []
            for index, image in enumerate(image_list):
                image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
                image = cv2.resize(image, _g.IMAGE_SHAPE[0:2])
                image = (image-_g.IMAGE_DEPTH/2)/_g.IMAGE_DEPTH
                image = image[np.newaxis, :]
                images.append(image)
            view = np.concatenate(images, axis=0)
            yield view, label

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32))
    dataset = dataset.shuffle(_g.TRAIN_BATCH_SIZE, _g.SEED)
    dataset = dataset.repeat()
    return dataset


def test_inputs():
    print()
    dataset = prepare_dataset(_g.TRAIN_LIST)
    print('shapes:', dataset.output_shapes)
    print('types:', dataset.output_types)
    data_it = dataset.make_one_shot_iterator()
    next_data = data_it.get_next()

    with tf.Session() as sess:
        for i in range(10):
            data = sess.run(next_data)
            print(len(data), data[0].shape, len(data[1]), np.min(data[0]), np.max(data[0]))


if __name__ == '__main__':
    test_inputs()
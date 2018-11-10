# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mt
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

import globals as _g


def load_list(list_file):
    image_list = np.loadtxt(list_file, dtype=str, skiprows=2)
    assert len(image_list) == _g.NUM_VIEWS
    return tf.constant(image_list)


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

    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(list_files), tf.constant(one_shot_labels)))
    dataset = dataset.map(parse_image, num_parallel_calls=mt.cpu_count())
    dataset = dataset.batch(_g.TRAIN_BATCH_SIZE)
    dataset = dataset.repeat()
    return dataset


def _read_py_function(filename, label):
    image_lists = np.loadtxt(filename.decode(), dtype=str, skiprows=2)
    assert len(image_lists) == _g.NUM_VIEWS
    images = [cv2.imread(image_name, cv2.IMREAD_UNCHANGED).astype(np.float32) for image_name in image_lists]
    # resize image
    resized_images = [cv2.resize(image, _g.IMAGE_SHAPE[0:2]) for image in images]
    # scale_image
    scaled_images = [(image - _g.IMAGE_DEPTH // 2) / _g.IMAGE_DEPTH for image in resized_images]
    # add axis
    axis_images = [image[np.newaxis, :] for image in scaled_images]
    # concat images
    image_decoded = np.concatenate(axis_images, axis=0)
    # convert type
    image_decoded.astype(np.float32)
    return image_decoded.astype(np.float32), label


def parse_image(image_file, label):
    return tf.py_func(_read_py_function, [image_file, label], [tf.float32, label.dtype])


def test_inputs():
    print()
    dataset = prepare_dataset(_g.TRAIN_LIST)
    print('shapes:', dataset.output_shapes)
    print('types:', dataset.output_types)
    data_it = dataset.make_one_shot_iterator()
    next_data = data_it.get_next()

    with tf.Session() as sess:
        for i in range(10):
            data, label = sess.run(next_data)
            print(len(data), len(label), data.shape, np.min(data), np.max(data))


if __name__ == '__main__':
    test_inputs()

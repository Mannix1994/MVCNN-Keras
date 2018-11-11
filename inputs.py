# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
import multiprocessing as mt
from tensorflow import keras

import globals as _g

_g.set_seed()


def prepare_dataset(path=''):
    """
    prepaer dataset using tf.data.Dataset
    :param path: the list file like data/train_lists_demo.txt
    and data/val_lists_demo.txt
    :return: a Dataset object
    """
    # read image list files name and labels
    lists_and_labels = np.loadtxt(path, dtype=str).tolist()
    # shuffle dataset
    np.random.shuffle(lists_and_labels)
    # split lists an labels
    list_files, labels = zip(*[(l[0], int(l[1])) for l in lists_and_labels])
    # one_shot encoding on labels
    one_shot_labels = keras.utils.to_categorical(labels, _g.NUM_CLASSES).astype(dtype=np.int32)
    # make data set
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(list_files), tf.constant(one_shot_labels)))
    # perform function parse_image on each pair of (data, label)
    dataset = dataset.map(parse_image, num_parallel_calls=mt.cpu_count())
    # set the batch size, Very important function!
    dataset = dataset.batch(_g.TRAIN_BATCH_SIZE)
    # repeat forever
    dataset = dataset.repeat()
    # compute steps_per_epoch
    steps_per_epoch = np.ceil(len(labels)/_g.TRAIN_BATCH_SIZE).astype(np.int32)
    return dataset, steps_per_epoch


def _read_py_function(filename, label):
    """
    read all NUM_VIEWS image(shape: (227, 227, 3)) file belong to one object
    and concat the to a 'View'(shape: (NUM_VIEWS, 227, 227, 3)). the shape of
    'View' is same as the MVCNN model's Inputs(file model.py, line 87).
    :param filename: a list file like data/airplane/test/1.txt
    :param label: model label
    :return: a 'View' and label
    """
    image_lists = np.loadtxt(filename.decode(), dtype=str, skiprows=2)
    # get NUM_VIEWS image
    image_lists = image_lists[:_g.NUM_VIEWS]
    # raise error
    if len(image_lists) != _g.NUM_VIEWS:
        raise ValueError('There haven\'t %d views in %s ' % (_g.NUM_VIEWS, filename))
    # read images
    images = [cv2.imread(image_name, cv2.IMREAD_UNCHANGED).astype(np.float32) for image_name in image_lists]
    # resize image to shape IMAGE_SHAPE
    resized_images = [cv2.resize(image, _g.IMAGE_SHAPE[0:2]) for image in images]
    # scale image from [0, 255] to [-0.5, 0.5]
    scaled_images = [(image - _g.IMAGE_DEPTH // 2) / _g.IMAGE_DEPTH for image in resized_images]
    # add axis
    axis_images = [image[np.newaxis, :] for image in scaled_images]
    # concat images
    view = np.concatenate(axis_images, axis=0)
    # convert type
    return view.astype(np.float32), label


def parse_image(filename, label):
    return tf.py_func(_read_py_function, [filename, label], [tf.float32, label.dtype])


def test_inputs():
    """
    test function prepare_dataset
    """
    dataset, steps = prepare_dataset(_g.VAL_LIST)
    print('shapes:', dataset.output_shapes)
    print('types:', dataset.output_types)
    print('steps:', steps)
    data_it = dataset.make_one_shot_iterator()
    next_data = data_it.get_next()

    with tf.Session() as sess:
        for i in range(10):
            data, label = sess.run(next_data)
            print(len(data), len(label), data.shape, np.min(data), np.max(data))


if __name__ == '__main__':
    test_inputs()

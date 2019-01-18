# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inputs
import model as m
import numpy as np
import config as _g
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from PIL import Image
import os

_g.set_seed()


def feature_image(path='data/M-PIE/train.txt'):
    # get model
    cnn1, fc8, _ = m.inference_multi_view()
    # load_weights
    cnn1.load_weights('model/cnn1.latest.weights.h5')
    fc8.load_weights('model/mvcnn.latest.weights.h5')
    # load analyse dataset
    ana_dataset, ana_steps = inputs.prepare_dataset(path, shuffle=False)
    print('shapes:', ana_dataset.output_shapes)
    print('types:', ana_dataset.output_types)
    print('steps:', ana_steps)
    # get a iterator of dataset
    data_it = ana_dataset.make_one_shot_iterator().get_next()
    # get default Session
    sess = K.get_session()

    # store default mvcnn predicts
    fc8_predicts = []
    for i in range(ana_steps):
        # get a batch of data
        views, label = sess.run(data_it)
        print(len(views), len(label), views.shape, np.min(views), np.max(views))
        # compute mvcnn's predicts
        fc8_predicts.append(fc8.predict(views))

        # for every view of views
        for idx, view in enumerate(views):
            # compute cnn1's predicts
            cnn1_predicts = cnn1.predict(view)
            print(cnn1_predicts.shape)
            # store the value's figure
            plt.figure(2)
            v_max = np.max(cnn1_predicts)
            v_min = np.min(cnn1_predicts)
            for index, v in enumerate(cnn1_predicts):
                x = np.linspace(1, v.size, v.size)
                plt.subplot(4, 3, index+1)
                plt.ylim((v_min, v_max))
                plt.plot(x, v.reshape(-1), lw=0.1)
            plt.tight_layout()
            plt.savefig('test/before_view_pool_%d_%d.png' % (i, idx))
            plt.clf()
            plt.close()

    # concat all fc8_predicts
    fc8_predicts = np.concatenate(fc8_predicts, axis=0)
    # draw a picture of predicts
    print(fc8_predicts.shape)
    data_size = len(fc8_predicts)
    v_max = np.max(fc8_predicts)
    v_min = np.min(fc8_predicts)
    for idx, fc8_predict in enumerate(fc8_predicts):
        plt.figure(1, [9, 2 * (data_size // 3 + 1)])
        x = np.linspace(1, fc8_predict.size, fc8_predict.size)
        plt.subplot(data_size // 3 + 1, 3, idx+1)
        plt.ylim((v_min, v_max))
        plt.plot(x, fc8_predict.reshape(-1))

    plt.tight_layout()
    plt.savefig('test/after_view_pool_%d.png')
    plt.clf()


def read_face(path):
    # read image list files name and labels
    lists_and_labels = np.loadtxt(path, dtype=str).tolist()
    # split lists an labels
    list_files, labels = zip(*[(l[0], int(l[1])) for l in lists_and_labels])
    for idx, lf in enumerate(list_files):
        image_lists = np.loadtxt(lf, dtype=str, skiprows=2)
        # get NUM_VIEWS image
        image_lists = image_lists[:_g.NUM_VIEWS]
        print(len(image_lists))
        # draw image
        plt.figure(1)
        for index, v in enumerate(image_lists):
            img = Image.open(v)
            plt.subplot(4, 3, index + 1)
            plt.imshow(img)
        plt.tight_layout()
        plt.savefig('test/src_%d.png' % idx)
        plt.clf()
        plt.close()


if __name__ == '__main__':
    feature_image()
    # read_face('data/M-PIE/train.txt')

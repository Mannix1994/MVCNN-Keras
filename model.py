# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.layers import *

import numpy as np
import tensorflow as tf
import globals as _g
from tensorflow.keras import backend as K

const_init = keras.initializers.constant(0)
xavier = keras.initializers.glorot_normal(seed=_g.SEED)
l2_reg = keras.regularizers.l2(0.004)


def _cnn1(input_shape):
    inputs = keras.Input(shape=input_shape, name='inputs')
    # this two lines didn't omit any parameter for showing how to define conv and pool layer
    conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu',
                   use_bias=True, kernel_initializer=xavier, name='conv1')(inputs)
    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1')(conv1)
    # we omit some default parameters
    conv2 = Conv2D(256, (5, 5), padding='same', activation='relu',
                   kernel_initializer=xavier, name='conv2')(pool1)
    pool2 = MaxPool2D((3, 3), (2, 2), name='pool2')(conv2)
    conv3 = Conv2D(384, (3, 3), padding='same', activation='relu',
                   kernel_initializer=xavier, name='conv3')(pool2)
    conv4 = Conv2D(384, (3, 3), padding='same', activation='relu',
                   kernel_initializer=xavier, name='conv4')(conv3)
    conv5 = Conv2D(256, (3, 3), padding='same', activation='relu',
                   kernel_initializer=xavier, name='conv5')(conv4)

    pool5 = MaxPool2D((3, 3), (2, 2), name='pool5')(conv5)

    reshape = Flatten(name='reshape')(pool5)

    cnn = keras.Model(inputs=inputs, outputs=reshape, name='cnn1')
    return cnn


def _split_inputs(inputs):
    slices = []
    for i in range(0, _g.NUM_VIEWS):
        slices.append(inputs[:, i, :, :, :])
    return slices


def _view_pool(views):
    expanded = [K.expand_dims(view, 0) for view in views]
    concated = K.concatenate(expanded, 0)
    reduced = K.max(concated, 0)
    return reduced


def inference_multi_view():

    inputs = Input(shape=_g.VIEWS_IMAGE_SHAPE, name='input')
    views = Lambda(_split_inputs, name='split')(inputs)

    view_pool = []
    cnn1_model = _cnn1(_g.IMAGE_SHAPE)
    # every view share the same cnn1_model(share the weights)
    for view in views:
        view_pool.append(cnn1_model(view))

    # view_pool
    # pool5_vp = vp.ViewPool(name='pool5_vp')(view_pool)
    pool5_vp = Lambda(_view_pool, name='view_pool')(view_pool)

    # cnn2 from here
    fc6 = Dense(units=4096, activation='relu', kernel_regularizer=l2_reg, name='fc6')(pool5_vp)
    dropout6 = Dropout(0.5, name='dropout6')(fc6)
    fc7 = Dense(4096, 'relu', kernel_regularizer=l2_reg, name='fc7')(dropout6)
    dropout7 = Dropout(0.5, name='dropout7')(fc7)
    fc8 = Dense(_g.NUM_CLASSES, 'softmax', kernel_regularizer=l2_reg, name='fc8')(dropout7)

    # softmax = Softmax(name='softmax')(fc8)

    model = keras.Model(inputs=inputs, outputs=fc8, name='MVCNN')
    return model


if __name__ == '__main__':
    mode = 2
    if mode == 1:
        # print cnn1's info
        cnn1_model = _cnn1(_g.IMAGE_SHAPE)
        keras.utils.plot_model(cnn1_model, to_file='model/cnn1_model.png', show_shapes=True)
        cnn1_model.summary()
    elif mode == 2:
        # print entire model's info
        model = inference_multi_view()
        keras.utils.plot_model(model, to_file='model/model.png', show_shapes=True)
        model.summary()
        model.save('mvcnn.model.h5')
    else:
        # load and print model info
        model = keras.models.load_model('mvcnn.model.h5')
        model.summary()


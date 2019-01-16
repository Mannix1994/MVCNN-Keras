# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import backend as K

import config as _g

_g.set_seed()

# define some constant initializer and regularizer
const_init = keras.initializers.constant(0)
xavier = keras.initializers.glorot_normal()
l2_reg = keras.regularizers.l2(0.004)


def _cnn1(input_shape):
    """
    this is the CNN1 Network in paper
    :param input_shape: a image's shape, not the shape of a batch of image
    :return: a model object
    """

    inputs = keras.Input(shape=input_shape, name='inputs')

    # this two layers don't omit any parameter for showing how to define conv and pool layer
    conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
                   padding='valid', activation='relu', use_bias=True,
                   kernel_initializer=xavier, name='conv1')(inputs)
    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid',
                      name='pool1')(conv1)

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
    """
    split inputs to NUM_VIEW input
    :param inputs: a Input with shape VIEWS_IMAGE_SHAPE
    :return: a list of inputs which shape is IMAGE_SHAPE
    """
    slices = []
    for i in range(0, _g.NUM_VIEWS):
        slices.append(inputs[:, i, :, :, :])
    return slices


def _view_pool(views):
    """
    this is the ViewPooling in the paper
    :param views: the NUM_VIEWS outputs of CNN1
    """
    expanded = [K.expand_dims(view, 0) for view in views]
    concated = K.concatenate(expanded, 0)
    reduced = K.max(concated, 0)
    return reduced


def inference_multi_view():
    """
    the Multi-View CNN in the paper
    :return: a keras model object
    """
    # input placeholder with shape (None, 12, 227, 227, 3)
    # 'None'=batch size; 12=NUM_VIEWS; (227, 227, 3)=IMAGE_SHAPE
    inputs = Input(shape=_g.VIEWS_IMAGE_SHAPE, name='input')

    # split inputs into views(a list), every element of
    # view has shape (None, 227, 227, 3).
    views = Lambda(_split_inputs, name='split')(inputs)

    # define a CNN1 model object
    cnn1_model = _cnn1(_g.IMAGE_SHAPE)

    view_pool = []
    # every view share the same cnn1_model(share the weights)
    for view in views:
        view_pool.append(cnn1_model(view))

    # view pool layer
    pool5_vp = Lambda(_view_pool, name='view_pool')(view_pool)

    # cnn2 from here
    # a full-connected layer
    fc6 = Dense(units=4096, activation='relu',
                kernel_regularizer=l2_reg, name='fc6')(pool5_vp)
    # a dropout layer, when call function evaluate and predict,
    # dropout layer will disabled automatically
    dropout6 = Dropout(0.6, name='dropout6')(fc6)

    fc7 = Dense(4096, 'relu', kernel_regularizer=l2_reg, name='fc7')(dropout6)
    dropout7 = Dropout(0.6, name='dropout7')(fc7)

    fc8 = Dense(_g.NUM_CLASSES, kernel_regularizer=l2_reg, name='fc8')(dropout7)

    softmax = Softmax(name='softmax')(fc8)

    mvcnn_model = keras.Model(inputs=inputs, outputs=softmax, name='MVCNN')
    return cnn1_model, mvcnn_model


if __name__ == '__main__':
    mode = 2
    if mode == 1:
        # print cnn1's info
        cnn1_model = _cnn1(_g.IMAGE_SHAPE)
        keras.utils.plot_model(cnn1_model, to_file='model/cnn1_model.png', show_shapes=True)
        cnn1_model.summary()
    elif mode == 2:
        # print entire model's info
        _, model = inference_multi_view()
        keras.utils.plot_model(model, to_file='model/model.png', show_shapes=True)
        model.summary()
        model.save('mvcnn.model.h5')
    else:
        # load and print model info
        model = keras.models.load_model('mvcnn.model.h5')
        model.summary()


# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inputs
import tensorflow as tf
import globals as _g

_g.set_seed()


if __name__ == '__main__':
    # prepare test dataset
    test_dataset, test_steps = inputs.prepare_dataset(_g.TEST_LIST)
    # load model
    model = tf.keras.models.load_model('model/latest.model.h5')
    # predict
    loss, accuracy = model.evaluate(test_dataset, steps=test_steps)
    print('test loss:', loss)
    print('test Accuracy:', accuracy)

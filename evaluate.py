# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inputs
import model as m
import tensorflow as tf
import config as _g

_g.set_seed()


if __name__ == '__main__':
    # prepare test dataset
    test_dataset, test_steps = inputs.prepare_dataset(_g.TEST_LIST)
    # get model
    model = m.inference_multi_view()
    # load_weights
    model.load_weights('model/latest.weights.h5')

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-5),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])

    # predict
    loss, accuracy = model.evaluate(test_dataset, steps=test_steps)
    print('test loss:', loss)
    print('test Accuracy:', accuracy)

# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inputs
import numpy as np
import tensorflow as tf
import globals as _g

_g.set_seed()


if __name__ == '__main__':
    # get a predict sample, with shape(12, 227, 227, 3)
    view, _ = inputs.read_and_process_image('data/airplane/test/1.txt', 0)
    # expand dim to (1, 12, 227, 227, 3)
    view = view[np.newaxis, :]
    print(view.shape)
    # get model
    model = tf.keras.models.load_model('model/latest.model.h5')
    # predict
    softmax = model.predict(view, 1)
    print(np.argmax(softmax))

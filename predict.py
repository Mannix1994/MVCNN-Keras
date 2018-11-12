# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras

import inputs
import model
import numpy as np
import globals as _g

_g.set_seed()


if __name__ == '__main__':
    # get a predict sample
    view, _ = inputs.read_and_process_image('data/airplane/test/1.txt', 0)
    # expand dim
    view = view[np.newaxis, :]
    print(view.shape)
    # get model
    model = model.inference_multi_view()
    # load_weights
    model.load_weights('model/latest.model.h5', by_name=True)
    # predict
    softmax = model.predict(view, 1)
    print(np.argmax(softmax))

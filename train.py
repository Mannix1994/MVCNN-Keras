# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras

import globals as _g
import model
import inputs


if __name__ == '__main__':
    train_dataset = inputs.prepare_dataset(_g.TRAIN_LIST)
    val_dataset = inputs.prepare_dataset(_g.VAL_LIST)
    model = model.inference_multi_view()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-4),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])
    model.fit(train_dataset, epochs=1, steps_per_epoch=100, validation_data=val_dataset, validation_steps=32)
    # test_pre_process()


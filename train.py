# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras

import globals as _g
import model
import inputs


def input_fn():
    return inputs.prepare_dataset(_g.TRAIN_LIST)


if __name__ == '__main__':
    train_dataset = inputs.prepare_dataset(_g.TRAIN_LIST)
    val_dataset = inputs.prepare_dataset(_g.VAL_LIST)

    train_with_multi_gpu = True
    model = model.inference_multi_view()

    if train_with_multi_gpu:
        model = keras.utils.multi_gpu_model(model, 2)

    model.compile(optimizer=keras.optimizers.Adam(lr=0.01, decay=1e-6),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])

    model.fit(train_dataset, epochs=100, steps_per_epoch=100,
              validation_data=val_dataset, validation_steps=32)


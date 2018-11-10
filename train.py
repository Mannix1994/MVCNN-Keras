# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras

import globals as _g
import model
import inputs
import pynvml


def get_gpu_count():
    pynvml.nvmlInit()
    gpu_num = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    return gpu_num


if __name__ == '__main__':
    gpu_num = get_gpu_count()

    train_dataset = inputs.prepare_dataset(_g.TRAIN_LIST)
    val_dataset = inputs.prepare_dataset(_g.VAL_LIST)

    train_with_multi_gpu = True
    model = model.inference_multi_view()

    if train_with_multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpu_num)

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-6),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])

    model.fit(train_dataset, epochs=100, steps_per_epoch=100,
              validation_data=val_dataset, validation_steps=32)

    model.save('model/latest.model.h5')



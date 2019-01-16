# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras

import model
import inputs
import pynvml
import argparse

import config as _g

_g.set_seed()


def get_gpu_count():
    """
    return the gpu number
    """
    pynvml.nvmlInit()
    gpu_number = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    return gpu_number


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--use-multi-gpu', action='store_true',
                        help='Using CUDA-enabled device to accelerate training model')
    arg, _ = parser.parse_known_args()

    train_with_multi_gpu = arg.use_multi_gpu

    gpu_num = get_gpu_count()

    # define train and validate dataset
    train_dataset, train_steps = inputs.prepare_dataset(_g.TRAIN_LIST)
    val_dataset, val_steps = inputs.prepare_dataset(_g.VAL_LIST)

    # define a MVCNN model
    cnn1, model = model.inference_multi_view()
    model.summary()
    model.load_weights('model/latest.weights.h5')

    origin_model = model
    if train_with_multi_gpu:
        # use the multi_gpu_model to train model with multi gpu
        print('Using GPU to train model')
        model = keras.utils.multi_gpu_model(model, gpu_num)

    # compile model. this is a multi-classification problem, so
    # the loss should be categorical_crossentropy.
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-5),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])

    # set callbacks
    callbacks = [
        # write TensorBoard' logs to directory 'logs'
        keras.callbacks.TensorBoard(log_dir='./logs'),
        # EarlyStopping for prevent overfitting
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2)
    ]

    # start training model
    model.fit(train_dataset, epochs=1, steps_per_epoch=train_steps,
              validation_data=val_dataset, validation_steps=val_steps, callbacks=callbacks)

    # save model's wights
    cnn1.save_weights('model/cnn1.latest.weights.h5', save_format='h5')
    origin_model.save_weights('model/mvcnn.latest.weights.h5', save_format='h5')

# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
import tensorflow as tf

import globals as _g
import model
import inputs


def input_fn():
    return inputs.prepare_dataset(_g.TRAIN_LIST)


if __name__ == '__main__':

    train_with_multi_gpu = True

    model = model.inference_multi_view()

    if not train_with_multi_gpu:
        train_dataset = inputs.prepare_dataset(_g.TRAIN_LIST)
        val_dataset = inputs.prepare_dataset(_g.VAL_LIST)

        model.compile(optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-4),
                      loss=keras.losses.categorical_crossentropy,
                      metrics=[keras.metrics.categorical_accuracy])

        model.fit(train_dataset, epochs=10, steps_per_epoch=100, validation_data=val_dataset, validation_steps=32)
    else:
        model.compile(optimizer=tf.train.AdamOptimizer())
        strategy = tf.contrib.distribute.MirroredStrategy()
        config = tf.estimator.RunConfig(train_distribute=strategy)
        keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, config=config, model_dir='./model_dir')
        keras_estimator.train(input_fn=input_fn, steps=10)

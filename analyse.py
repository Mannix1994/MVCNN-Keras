# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inputs
import model as m
import numpy as np
import config as _g
import tensorflow.keras.backend as K
_g.set_seed()
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # get model
    cnn1, model = m.inference_multi_view()
    # load_weights
    cnn1.load_weights('model/cnn1.latest.weights.h5')
    model.load_weights('model/mvcnn.latest.weights.h5')
    # predict
    # softmax = model.predict(view, 1)
    # print(np.argmax(softmax))

    # load analyse dataset
    ana_dataset, ana_steps = inputs.prepare_dataset('data/M-PIE/train.txt', shuffle=False)

    print('shapes:', ana_dataset.output_shapes)
    print('types:', ana_dataset.output_types)
    print('steps:', ana_steps)
    data_it = ana_dataset.make_one_shot_iterator().get_next()

    sess = K.get_session()
    mvcnn_predicts = []
    for i in range(ana_steps):
        views, label = sess.run(data_it)
        print(len(views), len(label), views.shape, np.min(views), np.max(views))
        # 计算cnn1的预测
        cnn1_predicts = []
        mvcnn_predicts.append(model.predict(views))
        for idx, view in enumerate(views):
            cnn1_predicts = cnn1.predict(view)
            print(cnn1_predicts.shape)
            plt.figure(2)
            for index, v in enumerate(cnn1_predicts):
                x = np.linspace(1, v.size, v.size)
                plt.subplot(4, 3, index+1)
                plt.plot(x, v.reshape(-1), marker='.', lw=0.1)
            plt.tight_layout()
            plt.savefig('test/before_view_pool_%d_%d.png' % (i, idx))
            plt.clf()

    mvcnn_predicts = np.concatenate(mvcnn_predicts, axis=0)
    print(mvcnn_predicts.shape)
    data_size = len(mvcnn_predicts)
    for idx, mvcnn_predict in enumerate(mvcnn_predicts):
        plt.figure(1, [9, 2 * (data_size // 3 + 1)])
        x = np.linspace(1, mvcnn_predict.size, mvcnn_predict.size)
        plt.subplot(data_size // 3 + 1, 3, idx+1)
        plt.plot(x, mvcnn_predict.reshape(-1), marker='.')

    plt.tight_layout()
    plt.savefig('test/after_view_pool_%d.png')
    plt.clf()

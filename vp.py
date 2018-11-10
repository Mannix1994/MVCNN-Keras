# coding=utf-8
# for python3
# this file just for a custom layer demo
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class ReduceMax(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ReduceMax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReduceMax, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        vp = K.max(x, 0)
        return vp

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class ViewPool(Layer):
    """ViewPool in MVCNN

    Arguments:
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(ViewPool, self).__init__(**kwargs)
        self.axis = 0

    def build(self, input_shape):
        # Used purely for shape validation.
        print(input_shape)
        if not isinstance(input_shape, list) or len(input_shape) < 1:
            raise ValueError('A `ViewPool` layer should be called '
                             'on a list of at least 1 inputs')
        if all([shape is None for shape in input_shape]):
            return
        reduced_inputs_shapes = [shape.as_list() for shape in input_shape]
        print(reduced_inputs_shapes)
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) > 1:
            raise ValueError('A `ViewPool` layer requires '
                             'inputs with matching shapes. '
                             'Got inputs shapes: %s' % (input_shape))

    def call(self, views):
        expanded = [K.expand_dims(view, self.axis) for view in views]
        concated = K.concatenate(expanded, self.axis)
        reduced = K.max(concated, self.axis)
        return reduced

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `ViewPool` layer should be called '
                             'on a list of inputs.')
        print(list(input_shape[0]))
        output_shape = input_shape[0]
        return tuple(output_shape)

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(ViewPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def from_config(cls, config):
        return cls(**config)

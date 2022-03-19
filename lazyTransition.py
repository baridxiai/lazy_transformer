# coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.utils import cka



class LT(tf.keras.layers.Layer):
    def __init__(self, layer, dropout):
        """
    Args:
        lyaer:
        dropout: dropout rate inside transition for training.
    """
        super(LT, self).__init__()
        # layer is the UT block. So pass the "call" function here.
        # e.g., self.layer = Transformer.call
        self.layer = layer
        self.dropout = dropout

    def build(self, input_shape):
        """Builds the layer."""
        self.num_units = input_shape[-1]
        self.halting_pro = 0
        self.pre_step = None
        super(LT, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        with tf.name_scope("Lazy"):
            x_step = self.layer(x, *args, **kwargs)
            self.halting_pro = tf.stop_gradient(cka.feature_space_linear_cka(x, x_step))
            # could be used for deep model, etc..
            # if training:
                # x_step = tf.nn.dropout(x_step, self.dropout)
                # x = tf.nn.dropout(x, self.dropout)
            y = x + (1 - self.halting_pro) * x_step
            return y, x_step

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/8 20:50
@Author     : Andy
@Email      : zd18zd@163.com
"""

import tensorflow as tf
from tensorflow import flags
from model.models import BaseModel

FLAGS = flags.FLAGS


flags.DEFINE_integer("lstm_cells", 128, "Number of LSTM cells.")
flags.DEFINE_integer("num_lstm_layers", 2, "Number of LSTM layers.")


class LSTMModel(BaseModel):
    def create_model(self, stk_model_input, is_training, **unused_params):
        """
        lstm layer
        :param stk_model_input: shape [batch_size, max_frames, feature_dim]
        :param is_training:
        :param unused_params:
        :return: last step output of lstm
        """
        lstm_size = FLAGS.lstm_cells
        num_lstm_layers = FLAGS.lstm_layers

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(num_lstm_layers)
            ])

        outputs, state = tf.nn.dynamic_rnn(cell=stacked_lstm,
                                           inputs=stk_model_input,
                                           dtype=tf.float32)

        return state[-1].h

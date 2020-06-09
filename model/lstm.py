#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/8 20:50
@Author     : Andy
@Email      : zd18zd@163.com
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import flags
from model.models import BaseModel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_lstm_layer", 3, "Number of LSTM layers.")
# 在这里定义了，在cnn & attention 里不能再定义，FLAGS会报错
flags.DEFINE_float("drop_rate", 0.5, "dropout ratio after LSTM. ")
flags.DEFINE_float("l2_penalty", 1e-8, "the l2 penalty of classifier weights and bias")


class LSTMModel(BaseModel):

    def __init__(self, input_window, feat_dim, hidden_size, n_output=1,
                 drop_rate=None, l2_penalty=None):
        self.input_window = input_window
        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate or FLAGS.drop_rate
        self.n_output = n_output
        self.l2_penalty = l2_penalty or FLAGS.l2_penalty

    def create_model(self, stk_input, stk_label, is_training, **unused_params):
        """
        lstm layer
        :param stk_input: shape [batch_size, max_frames, feature_dim]
        :param stk_label: shape [batch_size]
        :param is_training:
        :param unused_params:
        :return: last step output of lstm
        """

        num_lstm_layer = FLAGS.num_lstm_layer
        print("{} lstm layer. ".format(num_lstm_layer))

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    self.hidden_size, forget_bias=1.0)
                for _ in range(num_lstm_layer)
            ])

        lstm_input = tf.reshape(stk_input, shape=[-1, self.input_window, self.feat_dim])
        _, state = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=lstm_input, dtype=tf.float32)
        lstm_out = state[-1].h

        print("lstm out:")
        print(lstm_out)

        if self.drop_rate > 0.:
            lstm_out = slim.dropout(lstm_out, keep_prob=1. - self.drop_rate,
                                    is_training=is_training, scope="lstm_dropout")

        hidden = slim.fully_connected(lstm_out, self.hidden_size, scope="lstm_hidden")
        hidden = slim.batch_norm(
            hidden,
            center=True,
            scale=True,
            is_training=is_training,
            scope="lstm_hidden_bn",
            fused=False)

        logits = slim.fully_connected(hidden, self.n_output, activation_fn=None,
                                      weights_regularizer=slim.l2_regularizer(self.l2_penalty),
                                      biases_regularizer=slim.l2_regularizer(self.l2_penalty),
                                      scope="lstm_logits")
        predictions = tf.nn.sigmoid(logits)

        labels = tf.cast(stk_label, tf.float32)
        logits = tf.reshape(logits, [-1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits
        ))

        print("lstm logits: {}".format(logits))
        # return {"stk_embedding": hidden, "logits": logits, "predictions": predictions, "loss": loss}
        return {"logits": logits, "predictions": predictions, "loss": loss}


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/8 21:05
@Author     : Andy
@Email      : zd18zd@163.com
"""


import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import flags

from model.models import BaseModel
from model.attention_layer import SelfAttention

FLAGS = flags.FLAGS


flags.DEFINE_integer("num_att_layer", 3, "Number of attention layer. ")
flags.DEFINE_integer("num_heads", 4, "Attention Encoder heads. ")


class ATTEncoder(BaseModel):

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
        :param stk_input: shape [batch_size, max_frames, feature_dim]
        :param stk_label: shape [batch_size]
        :param is_training:
        :param unused_params:
        :return:
        """
        num_att_layer = FLAGS.num_att_layer
        att_layers = []
        for i in range(num_att_layer):
            print("{} attention layer .".format(i+1))
            self_att = SelfAttention(self.hidden_size, FLAGS.num_heads,
                                     FLAGS.drop_rate, is_training)
            att_layers.append(self_att)

        att_input = tf.reshape(stk_input, shape=[-1, self.input_window, self.feat_dim])
        batch_size = tf.shape(att_input)[0]
        padding_bias = tf.zeros([batch_size, 1, 1, self.input_window])

        # multi layer
        att_out = att_input
        for self_att in att_layers:
            att_out = self_att(att_out, padding_bias)

        print("att out: ")
        print(att_out)

        att_out = tf.reduce_mean(att_out, axis=1)

        print("att out after reduce mean: ")
        print(att_out)

        if self.drop_rate > 0.:
            att_out = slim.dropout(att_out, keep_prob=1. - self.drop_rate,
                                   is_training=is_training, scope="att_dropout")

        hidden = slim.fully_connected(att_out, self.hidden_size, scope="att_hidden")
        hidden = slim.batch_norm(
            hidden,
            center=True,
            scale=True,
            is_training=is_training,
            scope="att_hidden_bn",
            fused=False)

        logits = slim.fully_connected(hidden, self.n_output, activation_fn=None,
                                      weights_regularizer=slim.l2_regularizer(self.l2_penalty),
                                      biases_regularizer=slim.l2_regularizer(self.l2_penalty),
                                      scope="att_logits")
        predictions = tf.nn.sigmoid(logits)

        labels = tf.cast(stk_label, tf.float32)
        logits = tf.reshape(logits, [-1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits
        ))
        print("attention logits: {}".format(logits))
        # return {"stk_embedding": hidden, "logits": logits, "predictions": predictions, "loss": loss}
        return {"logits": logits, "predictions": predictions, "loss": loss}


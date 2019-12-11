#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/8 21:05
@Author     : Andy
@Email      : zd18zd@163.com
"""


import tensorflow as tf
from tensorflow import flags
from model.models import BaseModel
from model.attention_layer import SelfAttention

FLAGS = flags.FLAGS


flags.DEFINE_integer("hidden_size", 128, "Attention Encoder hidden size. ")
flags.DEFINE_integer("num_heads", 4, "Attention Encoder heads. ")
flags.DEFINE_float("dropout", 0.5, "Attention Encoder dropout rate. ")


class ATTEncoder(BaseModel):
    def create_model(self, stk_model_input, is_training, **unused_params):
        """
        :param stk_model_input: shape [batch_size, max_frames, feature_dim]
        :param is_training:
        :param unused_params:
        :return:
        """
        self_att = SelfAttention(FLAGS.hidden_size, FLAGS.num_heads,
                                 FLAGS.dropout, is_training)
        padding_bias = tf.zeros_like(stk_model_input)
        return self_att(stk_model_input, padding_bias)

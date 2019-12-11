#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/8 20:52
@Author     : Andy
@Email      : zd18zd@163.com
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import flags
from model.models import BaseModel

FLAGS = flags.FLAGS


flags.DEFINE_integer("out_channel", 1, "CNN output channel. ")
flags.DEFINE_integer("kernel_height", 3, "CNN kernel size. ")
flags.DEFINE_integer("kernel_width", 8, "CNN kernel size. ")
flags.DEFINE_integer("strides", 1, "CNN strides. ")
flags.DEFINE_string("padding", "VALID", "CNN padding type. ")


class CNNModel(BaseModel):
    def create_model(self, stk_model_input, is_training, **unused_params):
        """
        conv layer
        :param stk_model_input: shape [batch_size, max_frames, feature_dim]
        :param is_training:
        :param unused_params:
        :return:
        """
        # shape [batch_size, max_frames, feature_dim, 1]
        cnn_input = tf.expand_dims(stk_model_input, -1)

        conv = slim.conv2d(cnn_input, FLAGS.out_channel, [FLAGS.kernel_height, FLAGS.kernel_width],
                           strides=FLAGS.strides, padding=FLAGS.padding)
        conv = tf.squeeze(conv)
        conv = tf.reduce_mean(conv, axis=1)
        return conv

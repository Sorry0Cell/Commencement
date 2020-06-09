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

flags.DEFINE_integer("num_cnn_layer", 3, "CNN layer. ")
flags.DEFINE_integer("out_channel", 1, "CNN output channel. ")
flags.DEFINE_integer("kernel_height", 3, "CNN kernel size. ")
flags.DEFINE_integer("kernel_width", 10, "CNN kernel size. ")
flags.DEFINE_integer("stride", 1, "CNN strides. ")
flags.DEFINE_string("padding", "SAME", "CNN padding type. ")


class CNNModel(BaseModel):

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
        conv layer
        :param stk_input: shape [batch_size, max_frames, feature_dim]
        :param stk_label: shape [batch_size]
        :param is_training:
        :param unused_params:
        :return:
        """
        # shape [batch_size, max_frames, feature_dim, 1]
        cnn_input = tf.reshape(stk_input, shape=[-1, self.input_window, self.feat_dim])
        conv = tf.expand_dims(cnn_input, -1)

        num_cnn_layer = FLAGS.num_cnn_layer
        for i in range(num_cnn_layer):
            print("{} cnn layer. ".format(i+1))
            with tf.variable_scope("cnn_layer_{}".format(i+1)):
                conv = slim.conv2d(conv, FLAGS.out_channel, [FLAGS.kernel_height, FLAGS.kernel_width],
                                   stride=FLAGS.stride, padding=FLAGS.padding)
                # 影响效果
                # conv = slim.batch_norm(
                #     conv, center=True, scale=True, is_training=is_training,
                #     scope="conv_bn_{}".format(i), fused=False)

        # shape [batch_size, max_frames, feature_dim]
        print("cnn out: ")
        print(conv)

        conv = tf.squeeze(conv, axis=-1)
        print("after squeeze: ")
        print(conv)

        cnn_out = tf.reduce_mean(conv, axis=1)

        print("after reduce_mean: ")
        print(cnn_out)

        if self.drop_rate > 0.:
            cnn_out = slim.dropout(cnn_out, keep_prob=1. - self.drop_rate,
                                   is_training=is_training, scope="cnn_dropout")

        hidden = slim.fully_connected(cnn_out, self.hidden_size,
                                      # weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      scope="cnn_hidden")

        hidden = slim.batch_norm(
            hidden,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cnn_hidden_bn",
            fused=False)

        logits = slim.fully_connected(hidden, self.n_output, activation_fn=None,
                                      weights_regularizer=slim.l2_regularizer(self.l2_penalty),
                                      biases_regularizer=slim.l2_regularizer(self.l2_penalty),
                                      # weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      scope="cnn_logits")

        predictions = tf.nn.sigmoid(logits)

        labels = tf.cast(stk_label, tf.float32)
        logits = tf.reshape(logits, [-1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits
        ))

        print("cnn logits: {}".format(logits))
        return {"logits": logits, "predictions": predictions, "loss": loss}

        # debug nan
        # return {"logits": logits, "predictions": predictions, "loss": loss,
        #         "hidden": hidden, "cnn_out": cnn_out, "cnn_input": cnn_input}

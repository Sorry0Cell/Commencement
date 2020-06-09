#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/12 13:33
@Author     : Andy
@Email      : zd18zd@163.com
"""


import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import flags
from model.models import BaseModel
from model import lstm, cnn, attention
from utils import find_class_by_name

FLAGS = flags.FLAGS

flags.DEFINE_string("mix_model_types", "LSTMModel,LSTMModel,LSTMModel", "Model of Each branch.")


class MixModel(BaseModel):

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
        mix model
        :param stk_input: shape [batch_size, max_frames, feature_dim]
        :param stk_label: shape [batch_size]
        :param is_training:
        :param unused_params:
        :return: last step output of lstm
        """
        mix_model_types = FLAGS.mix_model_types
        mix_model_types = mix_model_types.split(",")
        mix_number = len(mix_model_types)
        print("MixModel sub model num: {} .".format(mix_number))
        sub_model_results = []
        for x in mix_model_types:
            if x not in ["CNNModel", "LSTMModel", "ATTEncoder"]:
                raise ValueError("Mix model only support lstm,cnn,att, but {} found . ".format(x))
            sub_model = find_class_by_name(x, [lstm, cnn, attention])(
                self.input_window, self.feat_dim, self.hidden_size)
            sub_model_results.append(sub_model.create_model(
                stk_input, stk_label, is_training
            ))

        mix_model_input = tf.reshape(stk_input, shape=[-1, self.input_window, self.feat_dim])
        # shape [batch_size, feat_dim]
        mix_frame_mean = tf.reduce_mean(mix_model_input, 1)

        mix_frame_mean = slim.batch_norm(mix_frame_mean,
                                         center=True,
                                         scale=True,
                                         fused=True,
                                         is_training=is_training,
                                         scope="mix_frame_mean_bn")
        # shape [batch_size, mix_number]
        mix_weights = slim.fully_connected(mix_frame_mean, mix_number, activation_fn=None,
                                           scope="mix_weights")
        # shape [batch_size, mix_number]
        mix_weights = tf.nn.softmax(mix_weights, axis=-1)
        # shape [batch_size, mix_number, 1]
        mix_weights = tf.expand_dims(mix_weights, -1)

        aux_preds = [res["predictions"] for res in sub_model_results]

        sub_logits = [res["logits"] for res in sub_model_results]
        sub_logits = [tf.reshape(x, [-1, 1]) for x in sub_logits]
        # shape [batch_size, mix_number, 1]
        sub_logits = tf.stack(sub_logits, axis=1)
        # shape [batch_size, 1]
        logits = tf.reduce_sum(tf.multiply(mix_weights, sub_logits), axis=1)
        predictions = tf.nn.sigmoid(logits)

        labels = tf.cast(stk_label, tf.float32)
        logits = tf.reshape(logits, [-1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits
        ))

        print("mix logits: {}".format(logits))
        return {"logits": logits, "predictions": predictions, "loss": loss}


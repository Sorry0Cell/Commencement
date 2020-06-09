#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/8 21:06
@Author     : Andy
@Email      : zd18zd@163.com
"""


import os
import time
import logging
import numpy as np
import tensorflow as tf

from model import lstm, cnn, attention, mix
from readers import get_input_arr
from utils import find_class_by_name

import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile

from sklearn.metrics import accuracy_score


FLAGS = flags.FLAGS

# task flags.
flags.DEFINE_string("task", "StockPred_CNN", "Task name. ")
flags.DEFINE_string("train_dir", "./train_dir/cnn_test_layer_1417_3_10_5",
                    "The directory to save the checkpoint.")
flags.DEFINE_string("model", "CNNModel",
                    "Which architecture to use for the model. Models are defined in models.py.")
# data flags.
flags.DEFINE_string("train_data_pattern", "tmp/201417_train_5_1_1219.csv", "File glob for the training dataset. ")
flags.DEFINE_string("test_data_pattern", "tmp/201417_test_5_1_1219.csv", "File glob for the testing dataset. ")

flags.DEFINE_integer("input_window", 5, "How many days history data used. ")
flags.DEFINE_integer("feat_dim", 10, "Dimension of features. ")
flags.DEFINE_integer("vocab_size", 27, "Predict class num. ")

# model flags.
flags.DEFINE_bool("start_new_model", True,
                  "If set, this will not resume from a checkpoint and will instead create a new model instance.")
flags.DEFINE_integer("hidden_size", 128, "Attention Encoder hidden size. ")

# Training flags.
flags.DEFINE_integer("num_epochs", 20,
                     "How many passes to make over the dataset before halting training.")
flags.DEFINE_integer("batch_size", 128,
                     "How many examples to process per batch.")
flags.DEFINE_integer("test_interval", 2000,
                     "How many iterations to process a test.")
flags.DEFINE_integer("log_interval", 100,
                     "How many iterations to log during training.")
flags.DEFINE_float("regularization_penalty", 1,
                   "How much weight to give to the regularization loss (the label loss has a weight of 1).")
flags.DEFINE_float("base_learning_rate", 0.0001,
                   "Which learning rate to start with.")
flags.DEFINE_float("learning_rate_decay", 0.95,
                   "Learning rate decay factor to be applied every learning_rate_decay_steps.")
flags.DEFINE_float("learning_rate_decay_steps", 1000,
                   "Multiply current learning rate by learning_rate_decay every learning_rate_decay_steps.")

# other flags.
flags.DEFINE_string("optimizer", "AdamOptimizer", "What optimizer class to use.")
flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
flags.DEFINE_bool("log_device_placement", False,
                  "Whether to write the device on which every op will run into the logs on startup.")
flags.DEFINE_bool("allow_soft_placement", False,
                  "Whether to use other device if not GPU .")
flags.DEFINE_string("col_label", "label", "Column name of label in csv file.")
flags.DEFINE_string("col_date", "date", "Column name of date in csv file.")
flags.DEFINE_string("col_stk_id", "stk_id", "Column name of stk_id in csv file.")


def build_graph(model,
                input_window,
                feat_dim,
                optimizer_class=tf.train.AdamOptimizer,
                base_learning_rate=0.01,
                learning_rate_decay_steps=10000,
                learning_rate_decay=0.95,
                clip_gradient_norm=1.0):
    """
    Create Tensorflow graph.
    :param model: The core model (e.g. logistic or neural net). It should inherit from BaseModel.
    :param input_window: How many days history data used.
    :param feat_dim: Dimension of each day's data.
    :param optimizer_class: Which optimization algorithm to use.
    :param base_learning_rate: learning rate to initialize the optimizer with.
    :param learning_rate_decay_steps:
    :param learning_rate_decay:
    :param clip_gradient_norm: Magnitude of the gradient to clip to.
    :return:
    """

    model_stk_input = tf.placeholder(tf.float32, [None, input_window * feat_dim], name="model_stk_input")
    model_stk_label = tf.placeholder(tf.int32, [None, ], name="model_stk_label")
    is_training = tf.placeholder(tf.bool, name="is_training")

    global_step = tf.Variable(0, trainable=False, name="global_step")

    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step,
        learning_rate_decay_steps,
        learning_rate_decay,
        staircase=True)

    optimizer = optimizer_class(learning_rate)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        model_result = model.create_model(model_stk_input, model_stk_label, is_training)

        # model_stk_embedding = model_result["stk_embedding"]
        model_predictions = model_result["predictions"]
        model_logits = model_result["logits"]
        model_loss = model_result["loss"]

        # debug nan
        # model_hidden = model_result["hidden"]
        # model_cnn_out = model_result["cnn_out"]
        # model_cnn_input = model_result["cnn_input"]

        train_op = slim.learning.create_train_op(
            model_loss,
            optimizer,
            global_step=global_step,
            clip_gradient_norm=clip_gradient_norm)

        tf.add_to_collection("global_step", global_step)

        tf.add_to_collection("model_stk_input", model_stk_input)
        tf.add_to_collection("model_stk_label", model_stk_label)
        tf.add_to_collection("is_training", is_training)

        # tf.add_to_collection("stk_embedding", model_stk_embedding)
        tf.add_to_collection("model_predictions", model_predictions)
        tf.add_to_collection("model_logits", model_logits)
        tf.add_to_collection("model_loss", model_loss)

        tf.add_to_collection("train_op", train_op)

        # debug nan
        # tf.add_to_collection("model_hidden", model_hidden)
        # tf.add_to_collection("model_cnn_out", model_cnn_out)
        # tf.add_to_collection("model_cnn_input", model_cnn_input)


class Trainer(object):
    """A Trainer to train a Tensorflow graph."""

    def __init__(self, task, model, input_window, feat_dim, hidden_size, train_dir, epochs, batch_size,
                 log_device_placement=True, allow_soft_placement=True):
        """
        Creates a Trainer.
        :param task:
        :param model:
        :param input_window:
        :param feat_dim:
        :param hidden_size:
        :param train_dir:
        :param epochs:
        :param batch_size:
        :param log_device_placement:
        :param allow_soft_placement:
        """
        self.task = task
        self.model = model
        self.input_window = input_window
        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.train_dir = train_dir
        self.config = tf.ConfigProto(log_device_placement=log_device_placement,
                                     allow_soft_placement=allow_soft_placement)
        self.epochs = epochs
        self.batch_size = batch_size
        self.last_model_eval_precision = 0.0
        self.last_model_export_step = 0

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

    def build_model(self):
        """Find the model and build the graph."""

        optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])
        model = find_class_by_name(FLAGS.model, [lstm, cnn, attention, mix])(
            self.input_window, self.feat_dim, self.hidden_size)

        build_graph(model=model,
                    input_window=self.input_window,
                    feat_dim=self.feat_dim,
                    optimizer_class=optimizer_class,
                    base_learning_rate=FLAGS.base_learning_rate,
                    learning_rate_decay=FLAGS.learning_rate_decay,
                    learning_rate_decay_steps=FLAGS.learning_rate_decay_steps,
                    clip_gradient_norm=FLAGS.clip_gradient_norm)

        return tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=5)

    def get_meta_filename(self, start_new_model, train_dir):
        if start_new_model:
            logging.info("%s: Flag 'start_new_model' is set. Building a new model.", self.task)
            return None

        latest_checkpoint = tf.train.latest_checkpoint(train_dir)
        if not latest_checkpoint:
            logging.info("%s: No checkpoint file found. Building a new model.", self.task)
            return None

        meta_filename = latest_checkpoint + ".meta"
        if not gfile.Exists(meta_filename):
            logging.info("%s: No meta graph file found. Building a new model.", self.task)
            return None
        else:
            return meta_filename

    def recover_model(self, meta_filename):
        logging.info("%s: Restoring from meta graph file %s", self.task, meta_filename)
        return tf.train.import_meta_graph(meta_filename, clear_devices=True)

    def run(self, train_data_pattern, test_data_pattern=None, start_new_model=False,
            thresh=0.5, test_interval=500, log_interval=20):
        """
        Performs training on the currently defined Tensorflow graph.
        :param train_data_pattern:
        :param test_data_pattern:
        :param start_new_model:
        :param thresh: used to calc acc
        :param test_interval: how many steps to eval test dataset
        :param log_interval: how many steps to log during training
        :return:
        """
        meta_filename = self.get_meta_filename(start_new_model, self.train_dir)
        save_prefix = os.path.join(self.train_dir, self.task)
        with tf.Graph().as_default() as graph:
            if meta_filename:
                saver = self.recover_model(meta_filename)
            else:
                saver = self.build_model()

            global_step = tf.get_collection("global_step")[0]

            model_stk_input = tf.get_collection("model_stk_input")[0]
            model_stk_label = tf.get_collection("model_stk_label")[0]
            is_training = tf.get_collection("is_training")[0]

            # stk_embedding = tf.get_collection("stk_embedding")[0]
            model_predictions = tf.get_collection("model_predictions")[0]
            model_logits = tf.get_collection("model_logits")[0]
            model_loss = tf.get_collection("model_loss")[0]

            train_op = tf.get_collection("train_op")[0]

            # debug nan
            # model_hidden = tf.get_collection("model_hidden")[0]
            # model_cnn_out = tf.get_collection("model_cnn_out")[0]
            # model_cnn_input = tf.get_collection("model_cnn_input")[0]

            init_op = tf.global_variables_initializer()

            logging.info("%s: Starting session.", self.task)
            with tf.Session(config=self.config) as sess:

                if start_new_model:
                    sess.run(init_op)

                # sess.run(init_op)

                logging.info("%s: Entering training loop.", self.task)
                best_test_acc = 0.0
                train_since = time.time()
                for i in range(self.epochs):
                    # 加载整个train_set
                    train_data_gen = get_input_arr(data_pattern=train_data_pattern, batch_size=self.batch_size,
                                                   shuffle=True, is_training=True,
                                                   feat_num=self.input_window * self.feat_dim)
                    try:
                        while True:
                            train_data = next(train_data_gen)
                            train_batch_input, train_batch_label, _, _ = train_data
                            batch_since = time.time()

                            _, global_step_val, train_loss_val, train_logits_val, train_pred_val = sess.run(
                                [train_op, global_step, model_loss, model_logits, model_predictions],
                                feed_dict={model_stk_input: train_batch_input,
                                           model_stk_label: train_batch_label,
                                           is_training: True})
                            per_batch_time = time.time() - batch_since

                            if global_step_val % log_interval == 0:
                                # 大于阈值即为1, 否则为0
                                train_batch_pred = train_pred_val > thresh

                                # logging.info("train_cnn_input: {}".format(train_cnn_input))
                                # logging.info("train_cnn_out: {}".format(train_cnn_out))
                                # logging.info("train_hidden_val: {}".format(train_hidden_val))
                                # logging.info("train_logits_val: {}".format(train_logits_val))
                                # logging.info("train_pred_val: {}".format(train_pred_val))

                                batch_acc = accuracy_score(y_true=train_batch_label, y_pred=train_batch_pred)
                                logging.info(
                                    "{}: training_step {}, train acc {:.4f}, train loss {:.2f},  cost {:.2f} secs.".format(
                                        self.task, global_step_val, batch_acc, train_loss_val, per_batch_time
                                    ))

                            if test_data_pattern and global_step_val % test_interval == 0:
                                logging.info("{}: Measure on test set".format(self.task))
                                test_data_gen = get_input_arr(data_pattern=test_data_pattern,
                                                              batch_size=self.batch_size,
                                                              shuffle=False, is_training=False,
                                                              feat_num=self.input_window * self.feat_dim)
                                test_loss_all = []
                                test_pred_all = []
                                test_true_all = []
                                test_since = time.time()
                                try:
                                    while True:
                                        test_data = next(test_data_gen)
                                        test_batch_input, test_batch_label, _, _ = test_data
                                        test_loss_val, test_pred_val = sess.run(
                                            [model_loss, model_predictions],
                                            feed_dict={model_stk_input: test_batch_input,
                                                       model_stk_label: test_batch_label,
                                                       is_training: False})
                                        test_pred_val = test_pred_val > thresh
                                        test_loss_all.append(test_loss_val)
                                        test_pred_all.append(test_pred_val)
                                        test_true_all.append(test_batch_label)
                                except StopIteration:
                                    assert len(test_pred_all) == len(test_true_all)

                                    test_time = time.time() - test_since
                                    test_loss = np.array(test_loss_all).mean()
                                    test_pred_all = np.concatenate(test_pred_all)
                                    test_true_all = np.concatenate(test_true_all)

                                    logging.info("{}: total {} records in test dataset".format(
                                        self.task, test_pred_all.shape[0]))
                                    test_acc = accuracy_score(y_true=test_true_all, y_pred=test_pred_all)
                                    logging.info(
                                        "{}: training step {}, test acc {:.4f}, test loss {:.2f}, cost {:.2f} sec.".format(
                                            self.task, global_step_val, test_acc, test_loss, test_time))

                                    if test_acc > best_test_acc:
                                        best_test_acc = test_acc
                                        print("Save model .")
                                        saver.save(sess, save_path=save_prefix, global_step=global_step_val)
                                        self.last_model_export_step = global_step_val
                    except StopIteration:
                        logging.info("{}: Finished {} epochs training.".format(self.task, i + 1))

                # 保存最后train完的模型
                saver.save(sess, save_path=save_prefix, global_step=global_step_val)
                train_time = time.time() - train_since

        logging.info("{}: Exited training loop, cost {:.2f} sec .".format(self.task, train_time))


def main(_):
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

    logging.info("{}: Tensorflow version: {}.".format(FLAGS.task, tf.__version__))
    trainer = Trainer(FLAGS.task, FLAGS.model, FLAGS.input_window, FLAGS.feat_dim, FLAGS.hidden_size, FLAGS.train_dir,
                      FLAGS.num_epochs, FLAGS.batch_size, FLAGS.log_device_placement, FLAGS.allow_soft_placement)

    trainer.run(train_data_pattern=FLAGS.train_data_pattern,
                test_data_pattern=FLAGS.test_data_pattern,
                start_new_model=FLAGS.start_new_model,
                test_interval=FLAGS.test_interval,
                log_interval=FLAGS.log_interval)


if __name__ == "__main__":
    app.run()

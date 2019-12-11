#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/8 21:06
@Author     : Andy
@Email      : zd18zd@163.com
"""


import os
import json
import time

import numpy as np
import tensorflow as tf

import losses
import my_readers

from model import lstm, cnn, attention
from get_input import get_input_train_tensors, get_input_test_tensors

import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils

from sklearn.metrics import accuracy_score
from sklearn import metrics

FLAGS = flags.FLAGS

if __name__ == "__main__":
    # task flags.
    flags.DEFINE_string("task", "", "Task name. ")
    # data flags.
    flags.DEFINE_string("train_dir", "./train_dir",
                        "The directory to save the checkpoint.")
    flags.DEFINE_string("train_data_pattern", "",
                        "File glob for the training dataset. ")
    flags.DEFINE_string("test_data_pattern", "",
                        "File glob for the testing dataset. ")
    flags.DEFINE_integer("vocab_size", 27,
                         "Predict class num. ")

    # model flags.
    flags.DEFINE_string(
        "model", "LSTMModel",
        "Which architecture to use for the model. Models are defined in models.py.")
    flags.DEFINE_bool(
        "start_new_model", False,
        "If set, this will not resume from a checkpoint and will instead create a new model instance.")

    # Training flags.
    flags.DEFINE_integer("train_batch_size", 128,
                         "How many examples to process per batch for training.")
    flags.DEFINE_integer("test_batch_size", 128,
                         "How many examples to process per batch for testing.")
    flags.DEFINE_integer("test_interval", 100,
                         "How many iterations to process a test.")
    flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                        "Which loss function to use for training the model.")
    flags.DEFINE_float("regularization_penalty", 1,
                       "How much weight to give to the regularization loss (the label loss has a weight of 1).")
    flags.DEFINE_float("base_learning_rate", 0.001,
                       "Which learning rate to start with.")
    flags.DEFINE_float("learning_rate_decay", 0.9,
                       "Learning rate decay factor to be applied every learning_rate_decay_steps.")
    flags.DEFINE_float("learning_rate_decay_steps", 100000,
                       "Multiply current learning rate by learning_rate_decay every learning_rate_decay_steps.")
    flags.DEFINE_integer("num_epochs", 20, "How many passes to make over the dataset before halting training.")
    flags.DEFINE_integer("max_steps", None, "The maximum number of iterations of the training loop.")

    # other flags.
    flags.DEFINE_integer("num_readers", 1,
                         "How many threads to use for reading input files.")
    flags.DEFINE_string("optimizer", "AdamOptimizer", "What optimizer class to use.")
    flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
    flags.DEFINE_bool("log_device_placement", False,
                      "Whether to write the device on which every op will run into the logs on startup.")
    flags.DEFINE_bool("allow_soft_placement", False,
                      "Whether to use other device if not GPU .")


def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def build_graph(model,
                reader,
                train_data_pattern,
                test_data_pattern,
                label_loss_fn=losses.CrossEntropyLoss(),
                base_learning_rate=0.01,
                learning_rate_decay_steps=10000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None):
    """
    Creates the Tensorflow graph.
    :param model: The core model (e.g. logistic or neural net). It should inherit from BaseModel.
    :param reader: The data file reader. It should inherit from BaseReader.
    :param train_data_pattern: path to the train data files.
    :param test_data_pattern: path to the test data files.
    :param label_loss_fn: loss to apply to the model. It should inherit from BaseLoss.
    :param base_learning_rate: learning rate to initialize the optimizer with.
    :param learning_rate_decay_steps:
    :param learning_rate_decay:
    :param optimizer_class: Which optimization algorithm to use.
    :param clip_gradient_norm: Magnitude of the gradient to clip to.
    :param regularization_penalty: How much weight to give the regularization loss compared to the label loss.
    :param num_readers:
    :param num_epochs:
    :return:
    """

    global_step = tf.Variable(0, trainable=False, name="global_step")

    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step,
        learning_rate_decay_steps,
        learning_rate_decay,
        staircase=True)

    optimizer = optimizer_class(learning_rate)

    _, train_stk_input, train_stk_label = get_input_train_tensors(
        reader, train_data_pattern, batch_size=FLAGS.train_batch_size, num_readers=num_readers, num_epochs=num_epochs)

    _, test_stk_input, test_stk_label = get_input_test_tensors(
        reader, test_data_pattern, batch_size=FLAGS.test_batch_size, num_readers=1)

    train_stk_feature_dim = len(train_stk_input.get_shape()) - 1
    test_stk_feature_dim = len(test_stk_input.get_shape()) - 1

    assert train_stk_feature_dim == test_stk_feature_dim

    train_stk_model_input = tf.nn.l2_normalize(train_stk_input, train_stk_feature_dim)
    test_stk_model_input = tf.nn.l2_normalize(test_stk_input, test_stk_feature_dim)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        train_result = model.create_model(train_stk_model_input)
        test_result = model.create_model(test_stk_model_input)

        train_predictions = train_result["predictions"]
        test_predictions = test_result["predictions"]
        stk_embedding = test_result["stk_embedding"]

        if "loss" in train_result.keys():
            train_loss = train_result["loss"]
        else:
            train_loss = label_loss_fn.calculate_loss(train_predictions, train_stk_label)

        train_aux_loss = tf.constant(0.0)
        if "aux_predictions" in train_result.keys():
            for pred in train_result["aux_predictions"]:
                train_aux_loss += label_loss_fn.calculate_loss(pred, test_stk_label)

        if "regularization_loss" in train_result.keys():
            train_reg_loss = train_result["regularization_loss"]
        else:
            train_reg_loss = tf.constant(0.0)

        train_reg_losses = tf.losses.get_regularization_losses()
        if train_reg_losses:
            train_reg_loss += tf.add_n(train_reg_losses)

        if "loss" in test_result.keys():
            test_loss = test_result["loss"]
        else:
            test_loss = label_loss_fn.calculate_loss(test_predictions, test_stk_label)

        # A dependency to the train_op.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if "update_ops" in train_result.keys():
            update_ops += train_result["update_ops"]
        if update_ops:
            with tf.control_dependencies(update_ops):
                barrier = tf.no_op(name="gradient_barrier")
                with tf.control_dependencies([barrier]):
                    train_loss = tf.identity(train_loss)
                    train_aux_loss = tf.identity(train_aux_loss)

        # Incorporate the L2 weight penalties etc.
        train_final_loss = regularization_penalty * train_reg_loss + train_loss + train_aux_loss

        train_op = slim.learning.create_train_op(
            train_final_loss,
            optimizer,
            global_step=global_step,
            clip_gradient_norm=clip_gradient_norm)

        tf.add_to_collection("global_step", global_step)
        tf.add_to_collection("train_loss", train_loss)
        tf.add_to_collection("test_top_loss", test_loss)
        tf.add_to_collection("train_predictions", train_predictions)
        tf.add_to_collection("test_predictions", test_predictions)
        tf.add_to_collection("train_stk_input", train_stk_input)
        tf.add_to_collection("train_stk_model_input", train_stk_model_input)
        tf.add_to_collection("test_stk_input", test_stk_input)
        tf.add_to_collection("test_stk_model_input", test_stk_model_input)
        tf.add_to_collection("train_stk_label", tf.cast(train_stk_label, tf.float32))
        tf.add_to_collection("test_stk_label", tf.cast(test_stk_label, tf.float32))
        tf.add_to_collection("stk_embedding", stk_embedding)
        tf.add_to_collection("train_op", train_op)


class Trainer(object):
    """A Trainer to train a Tensorflow graph."""

    def __init__(self, task, train_dir, model, reader,
                 log_device_placement=True,
                 allow_soft_placement=True,
                 max_steps=None):
        """
        Creates a Trainer.
        :param task:
        :param train_dir:
        :param model:
        :param reader:
        :param log_device_placement:
        :param max_steps:
        """
        self.task = task
        self.train_dir = train_dir
        self.model = model
        self.reader = reader

        self.config = tf.ConfigProto(log_device_placement=log_device_placement,
                                     allow_soft_placement=allow_soft_placement)
        self.max_steps = max_steps
        self.max_steps_reached = False
        self.last_model_eval_precision = 0.0
        self.last_model_export_step = 0

    def run(self, start_new_model=False):
        """Performs training on the currently defined Tensorflow graph.

        Returns:
          A tuple of the training Hit@1 and the training PERR.
        """

        meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

        with tf.Graph().as_default() as graph:

            if meta_filename:
                saver = self.recover_model(meta_filename)
            else:
                saver = self.build_model(self.model, self.reader)

            global_step = tf.get_collection("global_step")[0]
            train_top_loss = tf.get_collection("train_loss")[0]
            test_top_loss = tf.get_collection("test_top_loss")[0]
            train_predictions = tf.get_collection("train_predictions")[0]
            test_predictions = tf.get_collection("test_predictions")[0]
            train_top_labels = tf.get_collection("train_top_labels")[0]
            test_top_labels = tf.get_collection("test_top_labels")[0]
            train_op = tf.get_collection("train_op")[0]
            init_op = tf.global_variables_initializer()

        logging.info("%s: Starting session.", self.task)
        with tf.Session(config=self.config) as sess:

            # try:
            if 1:
                logging.info("Entering training loop.")
                while not self.max_steps_reached:

                    batch_start_time = time.time()
                    _, global_step_val, train_top_loss_val, train_predictions_val, train_top_labels_val = sess.run(
                        [train_op, global_step, train_top_loss, train_predictions, train_top_labels])
                    seconds_per_batch = time.time() - batch_start_time
                    if self.max_steps and self.max_steps <= global_step_val:
                        self.max_steps_reached = True

                    if global_step_val % 10 == 0:
                        train_metric = utils.CalMetric(train_predictions_val, train_top_labels_val)
                        logging.info("%s: training step " + str(global_step_val) + " top_precision: " + (
                                    "%.4f" % train_metric['top_precision']) + " top_recall: " + (
                                                 "%.4f" % train_metric['top_recall']) + " top_f1_score: " + (
                                                 "%.4f" % train_metric['top_f1_score']) + " top_loss: " + str(
                            train_top_loss_val) + " cost: " + str(
                            seconds_per_batch), self.task)

                        if global_step_val % FLAGS.test_interval == 0:
                            batch_start_time = time.time()
                            global_step_val, test_top_loss_val, test_predictions_val, test_top_labels_val = sess.run(
                                [global_step, test_top_loss, test_predictions, test_top_labels])
                            seconds_per_batch = time.time() - batch_start_time
                            test_metric = utils.CalMetric(test_predictions_val, test_top_labels_val)

                            logging.info("%s: training step " + str(global_step_val) + \
                                         " test_top_precision: " + ("%.4f" % test_metric['top_precision']) + \
                                         " test_top_recall: " + ("%.4f" % test_metric['top_recall']) + \
                                         "test_top_f1_score: " + ("%.4f" % test_metric['top_f1_score']) + \
                                         " top_loss: " + str(train_top_loss_val) + " cost: " + str(seconds_per_batch),
                                         self.task)

                            time_to_export = (test_metric['top_f1_score'] - self.last_model_eval_precision) / (
                                    self.last_model_eval_precision + 0.0000001) > 0.001 \
                                             and np.abs(
                                test_metric['top_f1_score'] - self.last_model_eval_precision) > 0.1 \
                                             and np.abs(test_metric['top_f1_score']) > 0.65 \

                            if time_to_export:
                                saver.save(sess, self.train_dir, global_step=global_step_val)
                                self.last_model_export_step = global_step_val

            # except tf.errors.OutOfRangeError:
            #     logging.info("%s: Done training -- epoch limit reached.",
            #                  self.task)

        logging.info("%s: Exited training loop.", self.task)

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
        return tf.train.import_meta_graph(meta_filename)

    def build_model(self, model, reader):
        """Find the model and build the graph."""

        label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
        optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

        build_graph(model=model,
                    reader=reader,
                    optimizer_class=optimizer_class,
                    clip_gradient_norm=FLAGS.clip_gradient_norm,
                    train_data_pattern=FLAGS.train_data_pattern,
                    test_data_pattern=FLAGS.test_data_pattern,
                    label_loss_fn=label_loss_fn,
                    base_learning_rate=FLAGS.base_learning_rate,
                    learning_rate_decay=FLAGS.learning_rate_decay,
                    learning_rate_decay_steps=FLAGS.learning_rate_decay_steps,
                    regularization_penalty=FLAGS.regularization_penalty,
                    num_readers=FLAGS.num_readers,
                    num_epochs=FLAGS.num_epochs)

        return tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=5)


def get_reader(reader_type="stk"):
    if reader_type == "stk":
        return my_readers.StockReader()


def main(_):
    logging.set_verbosity(tf.logging.INFO)
    logging.info("Tensorflow version: {}.".format(tf.__version__))
    model = find_class_by_name(FLAGS.model, [lstm, cnn, attention])()

    reader = get_reader(FLAGS.vocab_size)


    Trainer(FLAGS.task, FLAGS.train_dir, model, reader,
            FLAGS.log_device_placement, FLAGS.allow_soft_placement,
            FLAGS.max_steps).run(start_new_model=FLAGS.start_new_model)


if __name__ == "__main__":
    app.run()

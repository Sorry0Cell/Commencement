#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/12 10:40
@Author     : Andy
@Email      : zd18zd@163.com
"""

import os
import time
import logging
import numpy as np
import tensorflow as tf

from readers import get_input_arr

from tensorflow import app
from tensorflow import flags

from sklearn.metrics import accuracy_score


FLAGS = flags.FLAGS

# task flags.
flags.DEFINE_string("task", "StockPred", "Task name. ")
# model flags.
flags.DEFINE_string("train_dir", "./train_dir/cnn_test_layer_1417_3_10_5/StockPred_CNN-2000",
                    "The directory to save the checkpoint.")

# data flags.
flags.DEFINE_string("test_data_pattern", "tmp/201417_test_5_1_1219.csv", "File glob for the testing dataset. ")
flags.DEFINE_integer("input_window", 5, "How many days history data used. ")
flags.DEFINE_integer("feat_dim", 10, "Dimension of features. ")
# flags.DEFINE_integer("vocab_size", 27, "Predict class num. ")

# Training flags.
flags.DEFINE_integer("batch_size", 256,
                     "How many examples to process per batch.")

# other flags.
flags.DEFINE_bool("log_device_placement", False,
                  "Whether to write the device on which every op will run into the logs on startup.")
flags.DEFINE_bool("allow_soft_placement", False,
                  "Whether to use other device if not GPU .")
flags.DEFINE_string("col_label", "label", "Column name of label in csv file.")
flags.DEFINE_string("col_date", "date", "Column name of date in csv file.")
flags.DEFINE_string("col_stk_id", "stk_id", "Column name of stk_id in csv file.")

flags.DEFINE_bool("search_best_thresh", True, "Whether to search best thresh. ")


def search_best_thresh(y_true, y_score, intervals):
    best_acc = 0.0
    best_interval = 0.0
    for x in intervals:
        y_pred = y_score > x
        tmp_acc = accuracy_score(y_pred=y_pred, y_true=y_true)
        print("{:.2f}: {:.4f}".format(x, tmp_acc))
        if tmp_acc > best_acc:
            best_interval = x
            best_acc = tmp_acc

    logging.info("best interval: {:.2f}, best acc: {:.4f} .".format(best_interval, best_acc))


def run_eval(test_data_pattern, checkpoint_dir, thresh=0.5,
             log_device_placement=True, allow_soft_placement=True):
    if os.path.isdir(checkpoint_dir):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    else:
        latest_checkpoint = FLAGS.train_dir
    meta_filename = latest_checkpoint + ".meta"

    if not meta_filename:
        raise ValueError("There is no checkpoint in {} . ".format(checkpoint_dir))
    with tf.Graph().as_default() as graph:

        logging.info("%s: Restoring from meta graph file %s", FLAGS.task, meta_filename)
        saver = tf.train.import_meta_graph(meta_filename, clear_devices=True)

        model_stk_input = tf.get_collection("model_stk_input")[0]
        model_stk_label = tf.get_collection("model_stk_label")[0]
        is_training = tf.get_collection("is_training")[0]

        # stk_embedding = tf.get_collection("stk_embedding")[0]
        model_predictions = tf.get_collection("model_predictions")[0]
        model_logits = tf.get_collection("model_logits")[0]
        model_loss = tf.get_collection("model_loss")[0]

        logging.info("%s: Starting session.", FLAGS.task)
        config = tf.ConfigProto(log_device_placement=log_device_placement,
                                allow_soft_placement=allow_soft_placement)
        with tf.Session(config=config) as sess:
            # 加载参数
            logging.info("{}: Restoring variables from {}.".format(FLAGS.task, latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

            logging.info("%s: Entering evaluation loop.", FLAGS.task)
            test_data_gen = get_input_arr(data_pattern=test_data_pattern,
                                          batch_size=FLAGS.batch_size,
                                          shuffle=False, is_training=False,
                                          feat_num=FLAGS.input_window * FLAGS.feat_dim)
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
                    test_loss_all.append(test_loss_val)
                    test_pred_all.append(test_pred_val)
                    test_true_all.append(test_batch_label)
            except StopIteration:
                test_time = time.time() - test_since
                test_loss = np.array(test_loss_all).mean()
                test_pred_all = np.concatenate(test_pred_all)
                test_true_all = np.concatenate(test_true_all)
                if FLAGS.search_best_thresh:
                    search_best_thresh(y_true=test_true_all, y_score=test_pred_all,
                                       intervals=list(np.arange(0.3, 0.8, 0.05)))
                else:
                    test_pred_all = test_pred_all > thresh
                    test_acc = accuracy_score(y_true=test_true_all, y_pred=test_pred_all)
                    logging.info("{}: test acc {:.4f}, test loss {:.2f}, cost {:.2f} sec.".format(
                        FLAGS.task, test_acc, test_loss, test_time))

    logging.info("%s: Exited evaluation loop.", FLAGS.task)


def main(_):
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    run_eval(test_data_pattern=FLAGS.test_data_pattern,
             checkpoint_dir=FLAGS.train_dir, thresh=0.5,
             log_device_placement=True, allow_soft_placement=True)


if __name__ == "__main__":
    app.run()

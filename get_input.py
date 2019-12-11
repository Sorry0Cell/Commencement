#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/9 15:14
@Author     : Andy
@Email      : zd18zd@163.com
"""

import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging


def get_input_train_tensors(reader,
                            data_pattern,
                            batch_size=128,
                            num_epochs=None,
                            num_readers=2):
    """
    Creates the section of the graph which reads the training data .
    :param reader: A class which parses the training data .
    :param data_pattern: A 'glob' style path to the data files .
    :param batch_size: How many examples to process at a time .
    :param num_epochs: How many passes to make over the training data. Set to 'None' to run indefinitely.
    :param num_readers: How many I/O threads to use .
    :return: A tuple containing the features tensor, labels tensor
    """
    logging.info("Using batch size of {} for training . ".format(batch_size))
    with tf.name_scope("train_input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find training files. data_pattern={} . ".format(data_pattern))
        logging.info("Number of training files: {}. ".format(len(files)))
        filename_queue = tf.train.string_input_producer(
            files, num_epochs=num_epochs, shuffle=True)
        train_data = [
            reader.prepare_reader(filename_queue, True) for _ in range(num_readers)
        ]

        return tf.train.shuffle_batch_join(
            train_data,
            batch_size=batch_size,
            capacity=batch_size * 5,
            min_after_dequeue=batch_size,
            allow_smaller_final_batch=True,
            enqueue_many=True)


def get_input_test_tensors(reader,
                           data_pattern,
                           batch_size=128,
                           num_readers=1):
    """
        Creates the section of the graph which reads the training data .
        :param reader: A class which parses the training data .
        :param data_pattern: A 'glob' style path to the data files .
        :param batch_size: How many examples to process at a time .
        :param num_readers: How many I/O threads to use .
        :return: A tuple containing the features tensor, labels tensor
    """
    logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
    with tf.name_scope("eval_input"):
        # files = [data_pattern]
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find the evaluation files.")
        filename_queue = tf.train.string_input_producer(
            files, shuffle=False, num_epochs=None)
        eval_data = [
            reader.prepare_reader(filename_queue, False) for _ in range(num_readers)
        ]
        return tf.train.batch_join(
            eval_data,
            batch_size=batch_size,
            capacity=batch_size * 3,
            allow_smaller_final_batch=True,
            enqueue_many=True)

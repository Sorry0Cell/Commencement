#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/11 10:30
@Author     : Andy
@Email      : zd18zd@163.com
"""

import numpy as np
import pandas as pd
import tensorflow as tf


def batch_generator(array_list, batch_size, shuffle=False):
    """
    :param array_list: containing both x & y
    :param batch_size:
    :param shuffle:
    :return:
    """
    total_num = array_list[0].shape[0]
    for temp_array in array_list:
        if temp_array.shape[0] != total_num:
            raise ValueError("the element in array_list should have same shape[0]")
    if shuffle:
        permutation_index = np.random.permutation(total_num)
        for i, temp_array in enumerate(array_list):
            array_list[i] = temp_array[permutation_index]
    batch_num = (total_num - 1) // batch_size + 1
    for i in range(batch_num):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, total_num)
        batch_list = []
        for temp_array in array_list:
            batch_list.append(temp_array[batch_start:batch_end])
        yield batch_list


class BaseReader(object):
    """Inherit from this class when implementing new readers."""
    def prepare_reader(self, unused_filename_queue):
        """Create a thread for generating prediction and label tensors. """
        raise NotImplementedError()


class StockReader(BaseReader):
    def prepare_reader(self, filename_queue, is_training):
        reader = tf.TextLineReader(skip_header_lines=1)

        key, value = reader.read(filename_queue)
        print(key)
        print(value)

        record_defaults = [[0] for _ in range(43)]
        feats_label_stk_date = tf.decode_csv(value, record_defaults=record_defaults)
        feats = feats_label_stk_date[:40]
        label = feats_label_stk_date[40]
        stk = feats_label_stk_date[41]
        date = feats_label_stk_date[42]
        return feats, label, stk, date


def main():
    f_path = "../tmp/2014_test_5_1.csv"
    fn_queue = tf.train.string_input_producer([f_path])
    stk_reader = StockReader()
    value = stk_reader.prepare_reader(fn_queue, True)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        val = sess.run(value)
        print(val)


if __name__ == '__main__':
    main()




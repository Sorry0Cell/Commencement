#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/11 12:30
@Author     : Andy
@Email      : zd18zd@163.com
"""

import numpy as np
import pandas as pd
from glob import glob
import logging
from utils import complete_stk_id


def batch_generator(array_list, batch_size, shuffle=False):
    """
    :param array_list: containing both x & y
    :param batch_size:
    :param shuffle:
    :return:
    """
    total_num = array_list[0].shape[0]
    for tmp_array in array_list:
        if tmp_array.shape[0] != total_num:
            raise ValueError("the element in array_list should have same shape[0]")
    if shuffle:
        permutation_index = np.random.permutation(total_num)
        for i, tmp_array in enumerate(array_list):
            array_list[i] = tmp_array[permutation_index]
    batch_num = (total_num - 1) // batch_size + 1
    for i in range(batch_num):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, total_num)
        batch_list = []
        for tmp_array in array_list:
            batch_list.append(tmp_array[batch_start:batch_end])
        yield batch_list


def read_csv(f_path):
    if not f_path.endswith(".csv"):
        raise ValueError("The input for read_csv should csv format but : {} . ".format(f_path))
    return pd.read_csv(f_path, index_col=False)


def get_data_gen_from_df(df, batch_size=32, shuffle=False, feat_num=40,
                         col_label="label", col_date="date", col_stk_id="stk_id"):
    """

    :param df:
    :param batch_size:
    :param shuffle:
    :param feat_num:
    :param col_label:
    :param col_date:
    :param col_stk_id:
    :return:
    """

    logging.info("Begin get_data_from_df . ")
    logging.info("feature columns: {} . ".format(str(df.columns)))
    col_feat = list(range(feat_num))
    col_feat = [str(x) for x in col_feat]
    feats = df[col_feat].values
    label = df[col_label].values
    date = df[col_date].values
    stk_id = df[col_stk_id].apply(complete_stk_id).values

    feats = feats.astype(np.float32)
    label = label.astype(np.int32)

    data_gen = batch_generator([feats, label, stk_id, date],
                               batch_size=batch_size, shuffle=shuffle)
    return data_gen


def get_input_arr(data_pattern, batch_size=32, shuffle=True, is_training=True,
                  feat_num=40, col_label="label", col_date="date", col_stk_id="stk_id"):
    """
    Creates the section of the graph which reads the training data .
    :param data_pattern: A 'glob' style path to the data files .
    :param batch_size:
    :param shuffle:
    :param is_training:
    :param feat_num: How many feats in each line.
    :param col_label:
    :param col_date:
    :param col_stk_id:
    :return: A tuple containing the features tensor, labels tensor
    """
    if is_training:
        logging.info("Using batch size of {} for training . ".format(batch_size))
    else:
        logging.info("Using batch size of {} for testing . ".format(batch_size))

    files = glob(data_pattern)
    if len(files) == 0:
        raise ValueError("Can't find files in {} . ".format(data_pattern))
    all_df = []
    for f in files:
        all_df.append(read_csv(f))
    if len(all_df) > 1:
        all_df = all_df[0].append(all_df[1:])
    else:
        all_df = all_df[0]

    logging.info("all_df shape before drop nan: {} . ".format(all_df.shape))
    all_df.dropna(inplace=True)
    logging.info("all_df shape after drop nan: {} . ".format(all_df.shape))

    return get_data_gen_from_df(all_df, batch_size=batch_size, shuffle=shuffle, feat_num=feat_num,
                                col_stk_id=col_stk_id, col_label=col_label, col_date=col_date)


def main():
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

    pattern = "tmp/2014_test_5_1.csv"
    data_gen = get_input_arr(data_pattern=pattern, batch_size=5, is_training=False)
    a = next(data_gen)
    print(len(a))
    for x in a:
        print(x.dtype)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/10 9:03
@Author     : Andy
@Email      : zd18zd@163.com
"""

import os
import time
import logging
import pickle as pkl
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import complete_stk_id

full_col = ['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'PB', 'PE', 'Volume', 'turnover_1']
feat_col = ['Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'PB', 'PE', 'Volume', 'turnover_1']


def std_scale_data(df):
    """
    standard normalization
    :param df:
    :return:
    """
    std_scaler = StandardScaler()
    return std_scaler.fit_transform(df[feat_col])


def min_max_scale_data(df):
    """
    min max normalization
    :param df:
    :return:
    """
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(df[feat_col])


def get_df(f_path, col_dt="Trddt", start_date=None, end_date=None):
    if not os.path.exists(args.csv_input):
        raise ValueError("The f_path for get_df doesn't exist: {} ".format(f_path))
    if not args.csv_input.endswith(".csv"):
        raise ValueError("The f_path for get_df should be csv format, but {} ".format(f_path))
    logging.info("Begin to get_df . ")
    df = pd.read_csv(f_path, index_col=False)
    if start_date is not None and end_date is not None:
        df["regular_date"] = df[col_dt].apply(lambda x: datetime.strptime(x, "%Y/%m/%d"))
        df = df.loc[(df["regular_date"] >= start_date)
                    & (df["regular_date"] <= end_date)]
    logging.info("df shape: {} . ".format(df.shape))
    return df


def norm_df(df, norm_type):
    """
    :param df:
    :param norm_type:
    :return:
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The df for norm_df should be pandas.DataFrame format, but: {} ".format(type(df)))
    if norm_type not in ["std", "min_max"]:
        raise ValueError("The norm_type for norm_df only support std or min_max, but: {} ".format(norm_type))
    if norm_type == "std":
        norm_arr = std_scale_data(df[feat_col])
    else:
        norm_arr = min_max_scale_data(df[feat_col])
    return norm_arr


def norm_by_stk_id(df, norm_type, col_stk_id="Stkcd", col_dt="Trddt"):
    """
    按股票，减去均值，除以方差
    :param df:
    :param norm_type:
    :param col_stk_id:
    :param col_dt:
    :return:
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The df for norm_by_stk_id should be pandas.DataFrame format, but: {} ".format(type(df)))
    if norm_type not in ["std", "min_max"]:
        raise ValueError("The norm_type for norm_by_stk_id only support std or min_max, but: {} ".format(norm_type))

    logging.info("Begin to norm_by_stk_id . ")
    logging.info("input_df shape: {} . ".format(df.shape))
    stk_ids = df[col_stk_id].value_counts().keys()
    stk_ids = list(stk_ids)

    res = {}
    for x in stk_ids:
        tmp_df = df.loc[df[col_stk_id] == x]
        tmp_arr = norm_df(tmp_df, norm_type)
        # 值都保存成 np.ndarray
        res[x] = {"date": tmp_df[col_dt].values, "data": tmp_arr}

    return res


def norm_all(df, norm_type, col_stk_id="Stkcd", col_dt="Trddt"):
    """
    减去整个数据集的均值，除以方差
    :param df:
    :param norm_type:
    :param col_stk_id:
    :param col_dt:
    :return:
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The df for norm_all should be pandas.DataFrame format, but: {} ".format(type(df)))
    if norm_type not in ["std", "min_max"]:
        raise ValueError("The norm_type for norm_all only support std or min_max, but: {} ".format(norm_type))

    logging.info("Begin to norm_all . ")
    logging.info("input_df shape: {} . ".format(df.shape))
    # 归一化
    norm_arr = norm_df(df, norm_type)
    logging.info("norm_arr shape: {} . ".format(norm_arr.shape))

    # 转成 DataFrame
    normalized_df = pd.DataFrame(norm_arr, columns=feat_col)
    logging.info("normalized_df shape: {} . ".format(normalized_df.shape))

    # TODO 为什么不加 tolist 会出错
    normalized_df[col_stk_id] = df[col_stk_id].tolist()
    normalized_df[col_dt] = df[col_dt].tolist()

    stk_ids = normalized_df[col_stk_id].value_counts().keys()
    stk_ids = list(stk_ids)

    a = normalized_df[col_stk_id].value_counts().keys()
    b = df[col_stk_id].value_counts().keys()

    print(a)
    print(b)

    logging.info("normalized_df stk_ids len: {} . ".format(len(normalized_df[col_stk_id].value_counts().keys())))
    logging.info("df stk_ids len: {} . ".format(len(df[col_stk_id].value_counts().keys())))

    res = {}
    for x in stk_ids:
        tmp_df = normalized_df.loc[normalized_df[col_stk_id] == x]
        # 值都保存成 np.ndarray
        if tmp_df.shape[0] == 0:
            raise ValueError("Wrong")
        res[x] = {"date": tmp_df[col_dt].values, "data": tmp_df[feat_col].values}

    return res


def merge_dfs(input_dic, col_stk_id="Stkcd", col_dt="Trddt"):
    """
    :param input_dic: key 是 stk_id, value 是一个dict(key是date, data)
    :param col_stk_id:
    :param col_dt:
    :return:
    """
    logging.info("Begin to merge_dfs .")
    total_stk_id = []
    total_date = []
    total_data = []
    for stk_id, date_data_dic in input_dic.items():
        tmp_date = date_data_dic["date"]
        tmp_data = date_data_dic["data"]
        total_stk_id += [stk_id] * tmp_date.shape[0]
        total_date.append(tmp_date)
        total_data.append(tmp_data)
    total_stk_id = np.array(total_stk_id)
    total_date = np.concatenate(total_date)
    total_data = np.concatenate(total_data)
    res_df = pd.DataFrame(total_data, columns=feat_col)
    res_df[col_stk_id] = total_stk_id
    res_df[col_dt] = total_date
    return res_df


def gen_model_data(df, input_window=7, label_window=1,
                   col_stk_id="Stkcd", col_dt="Trddt", col_clsp="Clsprc"):
    """
    生成模型的数据
    :param df:
    :param input_window: 根据几天的历史数据预测
    :param label_window: 比如 1-7 号作为input，那么第 7+label_window 的价格 减去第 7 天的作为label
    :param col_stk_id:
    :param col_dt:
    :param col_clsp: 用来计算label
    :return: dict, input_window（多少天的历史数据）, label_window（多少天计算label）, input_data（训练数据）,
    """

    stk_ids = df[col_stk_id].value_counts().keys()

    # 根据 stk_id 存储 训练/测试数据
    input_data = {}

    for stk_id in stk_ids:
        tmp_df = df.loc[df[col_stk_id] == stk_id]
        tmp_len = tmp_df.shape[0]
        if tmp_len-label_window+1 <= input_window:
            continue
        tmp_feat = tmp_df[feat_col].values
        tmp_clsp = tmp_df[col_clsp].values
        tmp_dt = tmp_df[col_dt].values

        stk_id_data = []
        stk_id_label = []
        stk_id_date = []

        for i in range(input_window, tmp_len-label_window+1):
            history = tmp_feat[i-input_window:i, :].reshape(1, -1)      # input: 过去 input_window 天的数据
            price_diff = tmp_clsp[i-1+label_window] - tmp_clsp[i-1]     # label: 根据收盘价是涨是跌来计算
            label = 1 if price_diff > 0 else 0
            date = tmp_dt[i]        # 保存日期
            stk_id_data.append(history)
            stk_id_label.append(label)
            stk_id_date.append(date)

        stk_id_data = np.concatenate(stk_id_data)
        stk_id_label = np.array(stk_id_label)
        stk_id_date = np.array(stk_id_date)

        stk_id_str = complete_stk_id(stk_id)
        input_data[stk_id_str] = {"data": stk_id_data, "label": stk_id_label, "date": stk_id_date}

    dataset = {"input_window": input_window, "label_window": label_window, "input_data": input_data}

    return dataset


def main(args, col_stk_id="Stkcd", col_dt="Trddt", save_dir="../tmp",
         save_norm_res=False, save_split_res=False,
         train_file_name="train_dataset.pkl",
         test_file_name="test_dataset.pkl"):
    """
    :param args:
    :param col_stk_id: 表示股票代码的列名
    :param col_dt: 表示日期的列名
    :param save_dir: 保存结果文件的目录
    :param save_norm_res: 是否保存归一化数据后生成的 csv
    :param save_split_res: 是否保存分割数据集后生成的 csv
    :param train_file_name:
    :param test_file_name:
    :return:
    """

    # 1) read data
    df = get_df(args.csv_input, start_date=args.data_date_start, end_date=args.data_date_end)
    # 2) normalize data
    if args.by_stk_id:
        res = norm_by_stk_id(df, args.norm_type)
    else:
        res = norm_all(df, args.norm_type)
    # 3) save norm data
    if save_norm_res:
        f_path = os.path.join(save_dir, "norm_res.pkl")
        with open(f_path, "wb") as f:
            pkl.dump(res, f)

    # 4) merge dfs
    merged_df = merge_dfs(res, col_stk_id=col_stk_id, col_dt=col_dt)
    merged_df["regular_date"] = merged_df[col_dt].apply(lambda x: datetime.strptime(x, "%Y/%m/%d"))
    logging.info("merged_df shape: {} . ".format(merged_df.shape))

    # 5) split data, according to time
    logging.info("Begin to split_data . ")
    train_date_start = datetime.strptime(args.train_date_start, "%Y/%m/%d")
    train_date_end = datetime.strptime(args.train_date_end, "%Y/%m/%d")
    train_df = merged_df.loc[(merged_df["regular_date"] >= train_date_start)
                             & (merged_df["regular_date"] <= train_date_end)]
    test_df = merged_df.loc[merged_df["regular_date"] > train_date_end]

    logging.info("train_df shape: {} . ".format(train_df.shape))
    logging.info(("test_df shape: {} . ".format(test_df.shape)))

    if save_split_res:
        f_path = os.path.join(save_dir, "train.csv")
        train_df.to_csv(f_path, index=False)
        f_path = os.path.join(save_dir, "test.csv")
        test_df.to_csv(f_path, index=False)

    # 6) generate model input data
    train_model_data = gen_model_data(train_df, args.input_window, args.label_window)
    test_model_data = gen_model_data(test_df, args.input_window, args.label_window)

    f_path = os.path.join(save_dir, train_file_name)
    with open(f_path, "wb") as f:
        pkl.dump(train_model_data, f)
    f_path = os.path.join(save_dir, test_file_name)
    with open(f_path, "wb") as f:
        pkl.dump(test_model_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_input", default="../data/lstm_data_201415_turnover.csv",
                        type=str, help="input data, csv format ")
    parser.add_argument("--norm_type", default="std", choices=["std", "min_max"],
                        type=str, help="how to normalize ")
    parser.add_argument("--by_stk_id", default=True,
                        type=bool, help="whether normalize according to stk_id or not ")
    parser.add_argument("--data_date_start", default="2014/1/1",
                        type=str, help="start date of data ")
    parser.add_argument("--data_date_end", default="2014/12/31",
                        type=str, help="end date of data ")
    parser.add_argument("--train_date_start", default="2014/1/1",
                        type=str, help="start date of training data ")
    parser.add_argument("--train_date_end", default="2014/9/30",
                        type=str, help="end date of training data ")
    parser.add_argument("--input_window", default=5,
                        type=int, help="history data length, for making input ")
    parser.add_argument("--label_window", default=1,
                        type=int, help="future data length, for making label")
    args = parser.parse_args()

    now_time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")

    log_file = "./MakeTrainingData_{}.log".format(now_time)
    # logging.basicConfig(filename=log_file, filemode="w",
    #                     format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    #                     datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

    since = time.time()

    main(args, save_norm_res=False, save_split_res=False,
         train_file_name="2014_train_{}_{}_by_stkid.pkl".format(args.input_window, args.label_window),
         test_file_name="2014_test_{}_{}_by_stkid.pkl".format(args.input_window, args.label_window))

    time_elapsed = time.time() - since
    logging.info("The code runs {:.0f} m {:.0f} s\n".format(time_elapsed // 60, time_elapsed % 60))

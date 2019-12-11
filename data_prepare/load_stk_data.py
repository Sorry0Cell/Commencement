#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/10 19:49
@Author     : Andy
@Email      : zd18zd@163.com
"""

import time
import logging
import argparse
import pickle as pkl
import numpy as np
import pandas as pd


def main(f_path="../tmp/2014_train_5_1.pkl"):
    with open(f_path, "rb") as f:
        dataset = pkl.load(f)

    print(dataset.keys())
    input_data = dataset["input_data"]
    print(len(input_data))
    for stk_id, data in input_data.items():
        print("{}: data shape {}, label_shape {}, date_shape {}".format(
            stk_id, data["data"].shape, data["label"].shape, data["date"].shape
        ))

    # print(input_data["601258"])


def trans_pkl_2_csv(pkl_path="../tmp/2014_train_5_1.pkl", csv_path="../tmp/2014_train_5_1.csv",
                    feat_len=40):
    """
    把 make_stk_data 生成的 .pkl 文件转成 csv
    :param pkl_path:
    :param csv_path:
    :param feat_len: 总共有多少特征
    :return:
    """
    if not pkl_path.endswith(".pkl"):
        raise ValueError("The pkl_path should endswith .pkl, but: {} . ".format(pkl_path))
    if not csv_path.endswith(".csv"):
        raise ValueError("The csv_path should endswith .csv, but: {} . ".format(csv_path))

    logging.info("Begin to read pkl file . \n")
    with open(pkl_path, "rb") as f:
        dataset = pkl.load(f)

    total_stk_id = []
    total_data = []
    total_label = []
    total_date = []

    logging.info("Begin to parse pkl file . ")
    input_data = dataset["input_data"]
    for stk_id, data in input_data.items():
        logging.info("{}: data shape {}, label_shape {}, date_shape {} . ".format(
            stk_id, data["data"].shape, data["label"].shape, data["date"].shape
        ))
        total_stk_id.append(np.array([stk_id] * data["data"].shape[0]))
        total_data.append(data["data"])
        total_date.append(data["date"])
        total_label.append(data["label"])

    total_stk_id = np.concatenate(total_stk_id)
    total_data = np.concatenate(total_data)
    total_date = np.concatenate(total_date)
    total_label = np.concatenate(total_label)

    df = pd.DataFrame(total_data, columns=list(range(feat_len)))
    df["label"] = total_label
    df["stk_id"] = total_stk_id
    df["date"] = total_date

    logging.info("Save csv file to : {} . \n".format(csv_path))
    df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_input", default="../tmp/2014_test_5_1.pkl",
                        type=str, help="input data, pkl format ")
    parser.add_argument("--csv_output", default="../tmp/2014_test_5_1.csv",
                        type=str, help="output data, csv format ")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    since = time.time()
    trans_pkl_2_csv(pkl_path=args.pkl_input, csv_path=args.csv_output)
    time_elapsed = time.time() - since
    logging.info("The code runs {:.0f} m {:.0f} s . \n".format(time_elapsed // 60, time_elapsed % 60))


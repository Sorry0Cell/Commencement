#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/18 10:01
@Author     : Andy
@Email      : zd18zd@163.com
"""


import os
import json
import time
import logging
import pandas as pd
from datetime import datetime


col_words = "cut_text"
col_stk = "stock"
col_stk_id = "stk_id"
col_dt = "date"
col_pos = "pos"
col_neg = "neg"
col_pos_num = "pos_num"
col_neg_num = "neg_num"
col_label = "label"

cut_news_senti_dir = "../data/news_cut_text_senti_first100"
save_dir = "../data/news_cut_text_senti_label_first_100"
news_total_path = "../data/news_cut_text_senti_label_all_first100.csv"


def main():
    """
    根据新闻正负情感词，标记label，并把所有的 csv 合并
    :return:
    """

    files = os.listdir(cut_news_senti_dir)
    total_df = []

    logging.info("{}: total files to process : {} . ".format(task, len(files)))
    for i, f in enumerate(files):
        logging.info("{}: Begin to process {} file . ".format(task, i+1))
        f_path = os.path.join(cut_news_senti_dir, f)
        tmp_df = pd.read_csv(open(f_path, encoding="gb18030"), index_col=False)
        logging.info("{}: tmp_df shape before filter : {}".format(task, tmp_df.shape))
        tmp_df = tmp_df.loc[(tmp_df[col_pos_num] != 0) & (tmp_df[col_neg_num] != 0)]
        logging.info("{}: tmp_df shape after filter : {}".format(task, tmp_df.shape))
        if tmp_df.shape[0] == 0:
            continue
        tmp_df[col_label] = (tmp_df[col_pos_num] > tmp_df[col_neg_num]).apply(int)
        logging.info("{}: tmp_df shape after label : {}".format(task, tmp_df.shape))

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        tmp_path = os.path.join(save_dir, f)
        # tmp_df.to_csv(tmp_path, encoding="gb18030", index=None)
        logging.info("{}: tmp_df save to : {}".format(task, tmp_path))

        total_df.append(tmp_df)

    # 合并成一个
    if len(total_df) == 0:
        total_df = total_df[0]
    else:
        total_df = total_df[0].append(total_df[1:])
    # total_df.to_csv(news_total_path,  encoding="gb18030", index=None)
    logging.info("{}: total_df save to : {}".format(task, news_total_path))


if __name__ == '__main__':
    now_time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    # log_file = "../data/mongo_search_{}.log".format(now_time)
    # logging.basicConfig(filename=log_file, filemode="w",
    #                     format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    #                     datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    task = "News Label"

    since = time.time()
    main()
    duration = time.time() - since
    logging.info("Finished, cost {:.2f} sec .".format(duration))


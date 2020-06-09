#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2020/2/17 14:34
@Author     : Andy
@Email      : zd18zd@163.com
"""


import os
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


save_dir = "../data/news_cut_text_senti_label_first_100"
news_total_path = "../data/news_total_20200217.csv"


def main():
    """
    根据新闻正负情感词，标记label，并把所有的 csv 合并
    :return:
    """

    files = os.listdir(save_dir)
    total_df = []

    logging.info("{}: total files to process : {} . ".format(task, len(files)))
    for i, f in enumerate(files):
        logging.info("{}: Begin to process {} file . ".format(task, i+1))
        f_path = os.path.join(save_dir, f)
        tmp_df = pd.read_csv(open(f_path, encoding="gb18030"), index_col=False)
        total_df.append(tmp_df)

    # 合并成一个
    if len(total_df) == 0:
        total_df = total_df[0]
    else:
        total_df = total_df[0].append(total_df[1:])

    # stat total num
    logging.info("{}: total_df shape : {}".format(task, total_df.shape))

    # stat pos num and neg num
    total_df[col_label] = total_df[col_label].apply(int)
    pos_df = total_df.loc[total_df[col_label] == 1]
    neg_df = total_df.loc[total_df[col_label] == 0]
    logging.info("{}: pos_df shape : {}".format(task, pos_df.shape))
    logging.info("{}: neg_df shape : {}".format(task, neg_df.shape))

    # total_df.to_csv(news_total_path,  encoding="gb18030", index=None)
    logging.info("{}: total_df save to : {}".format(task, news_total_path))


if __name__ == '__main__':
    now_time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    task = "News Stat Num"

    since = time.time()
    main()
    duration = time.time() - since
    logging.info("Finished, cost {:.2f} sec .".format(duration))


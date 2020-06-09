#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/17 15:30
@Author     : Andy
@Email      : zd18zd@163.com
"""

import os
import json
import time
import logging
import pandas as pd
from datetime import datetime
from utils import read_txt


POS_PATH = "../data/positive.txt"
NEG_PATH = "../data/negative.txt"
name_2_id = "../data/stk_name_id.json"

col_words = "cut_text"
col_stk = "stock"
col_stk_id = "stk_id"
col_dt = "date"
col_pos = "pos"
col_neg = "neg"
col_pos_num = "pos_num"
col_neg_num = "neg_num"
cut_news_dir = "../data/news_cut_text"
save_dir = "../data/news_cut_text_senti_first100"


def get_pos(cut_text):
    """
    统计分词结果中有哪些 pos 词汇
    :param cut_text:
    :return:
    """
    pos = set(read_txt(POS_PATH, encoding="gb18030"))
    res = []
    for x in cut_text[:100]:
        if x in pos:
            res.append(x)
    return res


def get_neg(cut_text):
    """
    统计分词结果中有哪些 neg 词汇
    :param cut_text:
    :return:
    """
    neg = set(read_txt(NEG_PATH, encoding="gb18030"))
    res = []
    for x in cut_text[:100]:
        if x in neg:
            res.append(x)
    return res


def stat_pos_neg(f_path, name_id_dic):
    """
    统计该文件中，每条新闻正面词汇、负面词汇的个数
    :param f_path:
    :param name_id_dic: 将名称转换成股票代码
    :return:
    """
    if not f_path.endswith(".csv"):
        raise ValueError("The f_path for stat_pos_neg should be .csv, but {}".format(f_path))
    logging.info("{}: stat_pos_neg, begin to process {} . ".format(task, f_path))
    df = pd.read_csv(open(f_path, encoding="gb18030"))

    tmp_serie = df[col_words].apply(eval)

    # 取每篇新闻的前100个词
    tmp_serie = tmp_serie.apply(lambda x: x[:100])
    df[col_words] = tmp_serie

    df[col_pos] = tmp_serie.apply(get_pos)
    df[col_pos_num] = df[col_pos].apply(len)

    df[col_neg] = tmp_serie.apply(get_neg)
    df[col_neg_num] = df[col_neg].apply(len)

    df[col_stk_id] = df[col_stk].apply(lambda x: name_id_dic.get(x, "empty"))
    df = df[[col_stk_id, col_stk, col_dt, col_words, col_pos, col_neg, col_pos_num, col_neg_num]]
    logging.info("{}: stat_pos_neg, res df shape {} . ".format(task, df.shape))

    f_name = f_path.split(os.sep)[-1]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, f_name)
    # df.to_csv(save_path, index=False, encoding="gb18030")
    logging.info("{}: stat_pos_neg, save to {} . ".format(task, save_path))


def main():
    """
    统计分词后的新闻中，正负情感词个数
    :return:
    """
    name2id_dic = json.load(open(name_2_id, "r"))

    files = os.listdir(cut_news_dir)
    logging.info("{}: total files to process : {} . ".format(task, len(files)))
    for i, f in enumerate(files):
        logging.info("{}: Begin to process {} file . ".format(task, i+1))
        f_path = os.path.join(cut_news_dir, f)
        stat_pos_neg(f_path, name2id_dic)


if __name__ == '__main__':
    now_time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    # log_file = "../data/mongo_search_{}.log".format(now_time)
    # logging.basicConfig(filename=log_file, filemode="w",
    #                     format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    #                     datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    task = "Stat Sentiment"

    since = time.time()
    main()
    duration = time.time() - since
    logging.info("Finished, cost {:.2f} sec .".format(duration))


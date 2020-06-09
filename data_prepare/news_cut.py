#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/15 13:16
@Author     : Andy
@Email      : zd18zd@163.com
"""

import re
import os
import jieba
import logging
import pandas as pd
from datetime import datetime
from utils import read_txt


def find_chinese(sent):
    pattern = re.compile(r"[^\u4e00-\u9fa5]")
    if isinstance(sent, str):
        return re.sub(pattern, "", sent)
    elif isinstance(sent, list):
        words = []
        for x in sent:
            res = re.sub(pattern, "", x)
            if len(res) > 0:
                words.append(res)
        return words
    else:
        raise ValueError("The input for find_chinese is str or list, but {} . ".format(type(sent)))


def get_extend_words():
    """
    get extend words, including senti-word and com-name
    :return:
    """
    pos = read_txt(POS_PATH, encoding="gb18030")
    neg = read_txt(NEG_PATH, encoding="gb18030")
    senti = pos + neg
    # too much
    # id_name_dic = json.load(open(id_2_name, "r"))
    # com_name = list(id_name_dic.values())
    com_name = os.listdir(news_dir)
    com_name = [x.split(".")[0] for x in com_name]

    extend_words = senti + com_name
    return extend_words


def cut_sent(sent):
    extend_words = get_extend_words()
    for x in extend_words:
        jieba.add_word(x)
    return list(jieba.lcut(sent))


def main():
    """
    中文分词，去除非中文
    :return:
    """
    news_files = os.listdir(news_dir)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for i, f in enumerate(news_files):
        logging.info("{}: Begin process {} file: {}".format(task, i+1, f))
        f_path = os.path.join(news_dir, f)
        tmp_df = pd.read_csv(open(f_path, encoding="gb18030"))
        # print(tmp_df.head())
        tmp_df["cut_text"] = tmp_df["text"].apply(cut_sent)
        tmp_df["cut_text"] = tmp_df["cut_text"].apply(find_chinese)
        logging.info("{}: shape: {}".format(task, tmp_df.shape))
        save_path = os.path.join(out_dir, f)
        # tmp_df.to_csv(save_path, index=False, encoding="gb18030")
        logging.info("{}: {} saved to {}".format(task, f, save_path))


if __name__ == '__main__':
    POS_PATH = "../data/positive.txt"
    NEG_PATH = "../data/negative.txt"
    id_2_name = "../data/stk_id_name.json"
    news_dir = "../data/news_1417"
    out_dir = "../data/news_cut_text"
    task = "News Cut"

    now_time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    # log_file = "../News_process_{}.log".format(now_time)
    # logging.basicConfig(filename=log_file, filemode="w",
    #                     format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    #                     datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

    main()


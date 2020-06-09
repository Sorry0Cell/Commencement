#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/18 15:28
@Author     : Andy
@Email      : zd18zd@163.com
"""


import os
import json
import time
import logging
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def read_emb(f_path):
    """
    读取 embedding
    :param f_path:
    :return: key 是 word, value 是 embedding
    """
    with open(f_path, "r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    emb_dic = {}
    for x in lines:
        x = x.split()
        word = x[0]
        emb = x[1:]
        emb_dic[word] = " ".join(emb)
    return emb_dic


def sent2emb(sent, emb_type):
    """
    把句子转成 embedding, sent2emb, avg pooling
    :param sent:
    :param emb_type:
    :return:
    """
    if emb_type not in ["glove", "senti_glove", "word2vec"]:
        raise ValueError("The emb_type for sent2emb only support glove or senti_glove . ")
    words = sent.split()
    if emb_type == "glove":
        words_emb = [glove_emb_dic.get(x, glove_unk) for x in words]

    elif emb_type == "senti_glove":
        words_emb = [senti_glove_emb_dic.get(x, senti_unk) for x in words]
    else:
        words_emb = [word2vec_emb_dic.get(x, word2vec_unk) for x in words]
    words_emb = [emb.split() for emb in words_emb]
    for i, emb in enumerate(words_emb):
        # print(len(emb))
        words_emb[i] = [eval(x) for x in emb]
    words_emb = np.array(words_emb)
    return np.mean(words_emb, axis=0)


def raw_data_2_emb(f_path, emb_type, log_per_step=50, save_to=None):
    """
    把原始数据集转成embedding，一条句子是一个embedding， oov 用unk表示
    :param f_path:
    :param emb_type:
    :param log_per_step:
    :param save_to:
    :return:
    """
    logging.info("{}: Begin to trans data to emb: {}, emb_type is {} .".format(task, f_path, emb_type))
    if emb_type not in ["glove", "senti_glove", "word2vec"]:
        raise ValueError("The emb_type for raw_data_2_emb only support glove or senti_glove . ")
    with open(f_path, "r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    data_feat = []
    logging.info("{}: Total {} sentences .".format(task, len(lines)))
    for i, sent in enumerate(lines):
        sent_emb = sent2emb(sent, emb_type)
        data_feat.append(sent_emb)
        if (i+1) % log_per_step == 0:
            logging.info("{}: Trans {} sentences to emb already . ".format(task, i+1))
            logging.info("{}: sent emb shape: {} . ".format(task, sent_emb.shape))

    res_data = np.array(data_feat)
    logging.info("{}: Finished trans data to emb, res_data shape: {} . ".format(task, res_data.shape))

    if save_to:
        logging.info("{}: dump res_data to: {} . ".format(task, save_to))
        with open(save_to, "wb") as f:
            pkl.dump(res_data, f)

    return res_data


def main():

    emb_type = "word2vec"
    pos_data = raw_data_2_emb(raw_pos_data, emb_type, save_to="../data/senti_task/pos_data_word2vec.pkl")
    raw_data = raw_data_2_emb(raw_neg_data, emb_type, save_to="../data/senti_task/neg_data_word2vec.pkl")


if __name__ == '__main__':
    task = "Senti Classification"
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

    GloVe_path = "../data/glove_emb/GloVe_vector200.txt"
    Senti_GloVe_path = "../data/glove_emb/SentiGloVe_vector200.txt"
    Word2Vec_path = "../data/glove_emb/Word2Vec_vector.txt"

    raw_pos_data = "../data/glove_senti_first100/pos_cut_text_senti_first100.txt"
    raw_neg_data = "../data/glove_senti_first100/neg_cut_text_senti_first100.txt"

    glove_emb_dic = read_emb(GloVe_path)
    senti_glove_emb_dic = read_emb(Senti_GloVe_path)
    word2vec_emb_dic = read_emb(Word2Vec_path)

    glove_unk = glove_emb_dic.get("<unk>")
    senti_unk = senti_glove_emb_dic.get("<unk>")
    word2vec_unk = word2vec_emb_dic.get("<unk>")

    since = time.time()

    main()

    time_elapsed = time.time() - since
    logging.info("The code runs {:.0f} m {:.0f} s\n".format(time_elapsed // 60, time_elapsed % 60))


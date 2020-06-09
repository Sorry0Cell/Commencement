#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/18 17:46
@Author     : Andy
@Email      : zd18zd@163.com
"""


import random
import numpy as np
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score


# random.seed(47)
# np.random.seed(47)

# random.seed(43)
# np.random.seed(43)


def load_pkl(f_path):
    if not f_path.endswith(".pkl"):
        raise ValueError("The input for load_pkl should endswith .pkl, but {} .".format(f_path))
    return pkl.load(open(f_path, "rb"))


def main():
    pos_data = load_pkl(pos_path)
    pos_label = np.ones([pos_data.shape[0]])
    neg_data = load_pkl(neg_path)
    neg_label = np.zeros([neg_data.shape[0]])
    total_data = np.concatenate([pos_data, neg_data], axis=0)
    total_label = np.concatenate([pos_label, neg_label], axis=0)
    x_train, x_val, y_train, y_eval = train_test_split(total_data, total_label, test_size=0.2)
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    y_eval_pred = log_reg.predict(x_val)

    f_score = f1_score(y_true=y_eval, y_pred=y_eval_pred)
    pre_score = precision_score(y_true=y_eval, y_pred=y_eval_pred)
    rec_score = recall_score(y_true=y_eval, y_pred=y_eval_pred)
    auc = roc_auc_score(y_true=y_eval, y_score=y_eval_pred)
    acc = accuracy_score(y_eval, y_eval_pred)

    print("precision: {:.4f}".format(pre_score))
    print("rec_score: {:.4f}".format(rec_score))
    print("f1score: {:.4f}".format(f_score))
    print("acc: {:.4f}".format(acc))
    print("auc: {:.4f}".format(auc))

    print(y_eval)
    print(y_eval_pred)

    print("pos num: {}".format(np.sum(y_eval)))
    print("pos shape: {}".format(np.shape(y_eval)))


if __name__ == '__main__':

    emb_type = "senti_glove"
    if emb_type == "senti_glove":
        pos_path = "./data/senti_task/pos_data_glove_senti.pkl"
        neg_path = "./data/senti_task/neg_data_glove_senti.pkl"
    elif emb_type == "glove":
        pos_path = "./data/senti_task/pos_data_glove.pkl"
        neg_path = "./data/senti_task/neg_data_glove.pkl"
    elif emb_type == "word2vec":
        pos_path = "./data/senti_task/pos_data_word2vec.pkl"
        neg_path = "./data/senti_task/neg_data_word2vec.pkl"
    else:
        raise ValueError("The emb_type only support glove or senti_glove . ")

    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/18 19:29
@Author     : Andy
@Email      : zd18zd@163.com
"""

import time
import numpy as np
from gensim.models import word2vec
from utils import read_txt


sentence_path = "../data/glove_senti_first100/all_phrase_first100.txt"


def train_word2vec(sentences, min_count=5, window=15, size=200):
    print("Begin to train word2vec . ")
    train_begin = time.time()

    model = word2vec.Word2Vec(sentences, min_count=min_count,
                              window=window, size=size)
    vocab_list = []
    emb_list = []
    for i, word in enumerate(model.wv.vocab):
        vocab_list.append(word)
        emb_list.append(model.wv[word])
    word_unk = "<unk>"
    emb_unk = np.array(emb_list).mean(axis=0)
    vocab_list.append(word_unk)
    emb_list.append(emb_unk)

    train_time = time.time() - train_begin

    print("Train word2vec runs {:.0f} m {:.0f} s .\n".format(train_time // 60, train_time % 60))

    return vocab_list, emb_list


def save_emb(vocab_list, emb_list, save_path="../data/glove_emb/Word2Vec_vector_tmp.txt"):
    print("Begin to write emb . ")

    save_begin = time.time()
    with open(save_path, "w") as f:
        for i in range(len(vocab_list)):
            word = vocab_list[i]
            emb = emb_list[i].tolist()
            emb = [str(x) for x in emb]
            emb = " ".join(emb)
            f.write("{} {}\n".format(word, emb))
            if (i + 1) % 200 == 0:
                print("write {} already ... ".format(i + 1))

        print("write {} already ... ".format(len(vocab_list)))

    duration = time.time() - save_begin
    print("Save emb runs {:.0f} m {:.0f} s\n".format(duration // 60, duration % 60))


def main():
    sentences = read_txt(sentence_path, encoding="gb18030")
    sentences = [x.split() for x in sentences]
    vocab_list, emb_list = train_word2vec(sentences)
    save_emb(vocab_list, emb_list)


if __name__ == '__main__':

    since = time.time()
    main()
    time_elapsed = time.time() - since

    print("The code runs {:.0f} m {:.0f} s\n".format(time_elapsed // 60, time_elapsed % 60))


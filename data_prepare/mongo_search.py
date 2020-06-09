#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import json
import time
import argparse
import logging
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from utils import complete_stk_id


news_db_ip = "10.220.138.247"


def change_to_time(item):
    """
    在数据库中，key为"date"的value是字符串，这个函数就是把字符串变为时间格式，后面才方便进行时间范围的选择
    :param item:搜索结果中的每一个元素进行转换
    :return: item
    """
    item["date"] = datetime.strptime(item["date"], "%Y年%m月%d日")
    return item


def drop_noise(string):
    pattern = re.compile("[\r\n\t\v\f\?\,]")
    return pattern.sub("", string.strip())


def work(item):
    item = change_to_time(item)
    item["text"] = drop_noise(item["text"])
    return item


def get_collection():
    """
    get db collection
    :return:
    """
    client = MongoClient(host=news_db_ip)
    db = client.store
    news = db.clean_news_300
    return news


def query_content(stock, s, e):
    """
    模糊查询：有关键词的text.以及指定的股票代码
    :param stock:股票代码
    :param s:开始时间
    :param e:结束时间
    :return:在有值的情况下，返回一个DataFrame
    """
    con_cursor = get_collection().find({"stock": stock})
    logging.info("{}—{}条相关的新闻".format(stock, con_cursor.count()))

    if con_cursor.count() > 0:
        result = [work(item) for item in con_cursor]
        # 把查询出来的结果转换为list,再储存为DataFrame
        df = pd.DataFrame(result)
        df_m = df[df["date"].isin(pd.date_range(start=s, end=e))]   # filter by date
        if len(df_m) > 0:
            df_m = df_m.loc[:, ["stock", "date", "_id", "text"]]     # output cols
            return df_m
        else:
            return pd.DataFrame([])
    else:
        return pd.DataFrame([])


def main(args):
    if not args.stk_id_file.endswith(".csv"):
        raise ValueError("The stk_id_file should endswith .csv . ")
    stk_id_df = pd.read_csv(args.stk_id_file, header=None)
    stk_ids = stk_id_df[0].tolist()
    stk_ids = [complete_stk_id(x) for x in stk_ids]

    stk_id_name_dict = json.load(open(args.stk_id_name_json, "r"))
    stk_names = [stk_id_name_dict.get(x, None) for x in stk_ids]
    stk_names = list(filter(lambda x: x is not None, stk_names))

    begin_date = datetime.strptime(args.begin_date, "%Y/%m/%d")
    end_date = datetime.strptime(args.end_date, "%Y/%m/%d")

    if not os.path.exists(args.news_dir):
        os.mkdir(args.news_dir)

    since = time.time()
    cnt = 0
    for x in stk_names:
        df = query_content(x, begin_date, end_date)
        logging.info("{}—总共的新闻条数: {}".format(x, df.shape[0]))

        if df.shape[0] > 0:
            logging.info("{}—不同的新闻天数: {}. ".format(
                x, df["date"].value_counts().shape[0]))
            f_path = os.path.join(args.news_dir, "{}.csv".format(x))
            df.to_csv(f_path, encoding="gb18030", index=False)
            logging.info("{}—第{}家, 保存至：{}".format(x, cnt, f_path))
            cnt += 1

    duration = time.time() - since
    logging.info("Finished, cost {:.2f} sec .".format(duration))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stk_id_file", default="../data/stk_id_201415.csv", type=str, help="The stock id file. ")
    parser.add_argument("--stk_id_name_json", default="../data/stk_id_name.json",
                        type=str, help="The stock id2name json file. ")
    parser.add_argument("--begin_date", default="2014/01/01",
                        type=str, help="The begin date of news. ")
    parser.add_argument("--end_date", default="2017/12/31",
                        type=str, help="The end date of news. ")
    parser.add_argument("--news_dir", default="../data/news_1417",
                        type=str, help="The dir to save news. ")
    args = parser.parse_args()

    now_time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    # log_file = "../data/mongo_search_{}.log".format(now_time)
    # logging.basicConfig(filename=log_file, filemode="w",
    #                     format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    #                     datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

    main(args)

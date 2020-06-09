#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time       : 2019/12/11 13:04
@Author     : Andy
@Email      : zd18zd@163.com
"""

import json
import logging


def read_txt(f_path, encoding="utf-8"):
    if not f_path.endswith(".txt"):
        raise ValueError("The f_path for read_txt should endswith .txt, but {} . ".format(f_path))
    with open(f_path, "r", encoding=encoding) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def complete_stk_id(x):
    """
    complete stk_id to be 6 length
    :param x:
    :return:
    """
    if isinstance(x, int):
        x = str(x)
    if len(x) < 6:
        return "0"*(6-len(x)) + x
    else:
        return x


def save_json(obj, f_path):
    """
    :param obj:
    :param f_path:
    :return:
    """
    if not isinstance(obj, dict) or not isinstance(f_path, str):
        raise ValueError("The input for save_json should be a dict and a str .")
    if not f_path.endswith(".json"):
        raise ValueError("The f_path for save_json should endswith .json")
    with open(f_path, "w") as f:
        json.dump(obj, f)


def save_json_file_v2(dict_list, f_prefix):
    """
    save net info of level 1/2/3
    :param dict_list:
    :param f_prefix:
    :return:
    """
    for i, level_dic in enumerate(dict_list):
        f_path = f_prefix + "{}.json".format(i+1)
        save_json(level_dic, f_path)
        logging.info("save json file to {}".format(f_path))
        logging.info("Successfully Save data to {}".format(f_path))


def save_txt_file(data_list, f_path):
    """
    save data list to f_path
    :param data_list:
    :param f_path:
    :return:
    """
    with open(f_path, "w") as f:
        for x in data_list:
            x = "\t".join(x)
            f.write("{}\n".format(x))
    logging.info("Successfully Save data list to {}".format(f_path))


def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def run_inner_try():
    """测试try 嵌套 try"""
    a = (x for x in range(6))
    try:
        while True:
            x = next(a)
            print("x: {}".format(x))
            b = (y for y in range(5))
            try:
                while True:
                    y = next(b)
                    print("y: {}".format(y))
            except StopIteration:
                print("Finished b")
    except StopIteration:
        print("Finished all")

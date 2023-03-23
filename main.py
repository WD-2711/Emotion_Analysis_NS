#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
# @Time  : 2023/03/21 19:47:08
# @Author: wd-2711
'''

from config import *
from trainer import *

if __name__ == "__main__":
    if args["mode"] == "train":
        k_fold(args)
        args["model_type"], args["loss"] = "bert_textcnn", 100000
        k_fold(args)
    elif args["mode"] == "test":
        tester(args)
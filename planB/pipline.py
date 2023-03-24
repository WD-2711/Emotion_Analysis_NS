#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
# @Time  : 2023/03/23 20:45:24
# @Author: wd-2711
'''

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    # 导入NLPCC14-SC数据集
    NS_train = pd.read_csv("./data/NLPCC14-SC/train.tsv", sep='\t')
    NS_test = pd.read_csv("./data/NLPCC14-SC/test.tsv", sep='\t')
    train = NS_train

    # 清除训练集中label为空的数据
    train['label'] = train['label'].fillna(-1)
    train = train[train['label']!=-1]
    train['label'] = train['label'].astype(int)

    # 填补句子中的空值
    train['text_a'] = train['text_a'].fillna('无')
    NS_test['text_a'] = NS_test['text_a'].fillna('无')
    NS_test['label'] = 0

    def replace_punctuation(example):
        '''替换英文字符'''
        example = list(example)
        pre = ''
        cur = ''
        for i in range(len(example)):
            if i == 0:
                pre = example[i]
                continue
            pre = example[i-1]
            cur = example[i]
            # [\u4e00-\u9fa5] 中文字符
            if re.match("[\u4e00-\u9fa5]", pre):
                if re.match("[\u4e00-\u9fa5]", cur):
                    continue
                elif cur == ',':
                    example[i] = '，'
                elif cur == '.':
                    example[i] = '。'
                elif cur == '?':
                    example[i] = '？'
                elif cur == ':':
                    example[i] = '：'
                elif cur == ';':
                    example[i] = '；'
                elif cur == '!':
                    example[i] = '！'
                elif cur == '"':
                    example[i] = '”'
                elif cur == "'":
                    example[i] = "’"
        return ''.join(example)

    # 替换英文字符 - 训练集
    rep_train = train['text_a'].map(replace_punctuation)
    rep_train = pd.concat([train['label'], rep_train], axis=1)
    train = rep_train
    # 替换英文字符 - NS测试集
    rep_test = NS_test['text_a'].map(replace_punctuation)
    rep_test = pd.concat([NS_test[['qid', 'label']], rep_test], axis = 1)
    NS_test = rep_test

    # 5 折分层抽样
    X = np.array(train.index)
    y = train['label'].to_numpy()
    def generate_data(random_state = 2020):
        '''5折抽样'''
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = random_state)
        i = 0
        for train_index, dev_index in skf.split(X, y):
            print("Fold:", i, "| 训练集大小:", len(train_index), "| 验证集大小:", len(dev_index))
            # 定义抽样结果的保存路径
            data_save_dir = "./data/task1_data_StratifiedKFold_{}/data_origin_{}/".format(random_state, i)
            if not os.path.exists(data_save_dir):
                os.makedirs(data_save_dir)
            # 获取抽样结果并保存它们
            tmp_train = train.iloc[train_index]
            tmp_dev = train.iloc[dev_index]
            tmp_train.to_csv(data_save_dir + "train.csv")
            tmp_dev.to_csv(data_save_dir + "dev.csv")
            # 在每折文件夹内都放入测试集，方便后续预测
            NS_test.to_csv(data_save_dir + 'NS_test.csv')
            i += 1

    generate_data(random_state = 2020)
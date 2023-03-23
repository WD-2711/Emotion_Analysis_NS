#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
# @Time  : 2023/03/21 19:18:38
# @Author: wd-2711

# @data_wash.py
  (1) Mainly carry out the statistics of sentence length and word frequency.
  (2) Perform data cleaning according to the abnormal value of the box plot of 
      the sentence length and save it in './data/train_wash.tsv'
'''
from config import *

def data_analysis(data_path:str, args:dict) -> tuple:
    """
        Statistics on data length and word frequency
    """

    data = pd.read_csv(data_path, sep='\t', header=0)
    # len analysis
    data['text_a_count'] = data['text_a'].str.len()
    # word analysis
    good_str = "".join([data.iloc[i, 1] for i in range(data.shape[0]) if data.iloc[i, 0] == 1])
    bad_str = "".join([data.iloc[i, 1] for i in range(data.shape[0]) if data.iloc[i, 0] == 0])
    def punc_del(d:list):
        return [dd for dd in d if dd not in args["punct"]]
    good_tokens, bad_tokens = punc_del(list(jieba.cut(good_str, cut_all=False))), punc_del(list(jieba.cut(bad_str, cut_all=False)))
    word_result = {
        "good":sorted(dict(Counter(good_tokens)).items(), key=lambda x:x[1], reverse=True),
        "bad":sorted(dict(Counter(bad_tokens)).items(), key=lambda x:x[1], reverse=True)
    }
    return (data, word_result)

def data_show(data:tuple, it:int, axs:np.ndarray) -> dict:
    """
        Display using boxplots and histograms
    """
    (df, wordR) = data

    # plot box image
    axs[it, 0].set_title('length analysis', fontsize=15)
    gl, bl = df.loc[df['label'] == 1]['text_a_count'].to_list(), df.loc[df['label'] == 0]['text_a_count'].to_list()
    ax = axs[it, 0].boxplot([gl, bl], labels=["good", "bad"])

    # plot bar
    axs[it, 1].set_title('good tokens', fontsize=15)
    vsum = sum([i[1] for i in wordR["good"]])
    x = [i[1]/vsum for i in wordR["good"]][:15]
    lab = [i[0] for i in wordR["good"]][:15]
    axs[it, 1].bar(range(len(x)), x, tick_label=lab)

    axs[it, 2].set_title('bad tokens', fontsize=15)
    vsum = sum([i[1] for i in wordR["bad"]])
    x = [i[1]/vsum for i in wordR["bad"]][:15]
    lab = [i[0] for i in wordR["bad"]][:15]
    axs[it, 2].bar(range(len(x)), x, tick_label=lab)

    # get length outliers and gate
    return {
        "good_fliers_num":len(ax['fliers'][0].get_data()[1]),
        "bad_fliers_num":len(ax['fliers'][1].get_data()[1]),
        "good_low_gate":ax['whiskers'][0].get_data()[1][0],
        "good_high_gate":ax['whiskers'][1].get_data()[1][1],
        "bad_low_gate":ax['whiskers'][2].get_data()[1][0],
        "bad_high_gate":ax['whiskers'][3].get_data()[1][1]
    }

def data_clean_save(clean_args:dict, data:pd.core.frame.DataFrame, it:int, args:dict):
    """
        Data cleaning and preservation
    """
    def clean(clean_args:dict, data:pd.core.frame.DataFrame, option:tuple):
        mask = (data['label'] == option[0]) & (data['text_a_count'].between(0, clean_args[option[1]]))
        total_num = data.loc[data['label'] == option[0]].shape[0]
        remove_num = total_num - data[mask].shape[0]
        print("[+] iter {} remove {:.2f}%({}) data of {}".format(it, (remove_num*100)/total_num, remove_num, option[1].split("_")[0]))
        return data[mask]
    
    good_data = clean(clean_args, data, (1, "good_high_gate"))
    bad_data = clean(clean_args, data, (0, "bad_high_gate"))
    data = pd.concat([good_data, bad_data])
    save_path = os.path.join(os.path.dirname(args["data_path"]), 'train_wash.tsv')
    data.to_csv(save_path, sep='\t', index=False)
    return save_path


if __name__ == "__main__":
    args = {
        # wash epoch
        "wash_num" : 5,
        # source data
        "data_path" : "./data/train.tsv", 
        # useless symbols
        "punct" : set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰ ︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？的了是我｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥~〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
    }

    # Iterative handling of outliers
    pp = args["data_path"]
    fig, axs = plt.subplots(args["wash_num"], 3, figsize=(30, 50))
    for it in range(args["wash_num"]):
        data_result = data_analysis(pp, args)
        clean_args = data_show(data_result, it, axs)
        pp = data_clean_save(clean_args, data_result[0], it, args)
    plt.show()
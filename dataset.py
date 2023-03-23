#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
# @Time  : 2023/03/21 19:38:35
# @Author: wd-2711
'''

from config import *

# data reading
def data_reader(args:dict, with_labels:bool=True) -> list:
  model_name = args["model_name"]
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  if with_labels:
    data_path = args["data_path"]
    data = pd.read_csv(data_path, sep='\t', header=0)
    data = data.sample(frac=1)
    sentences = data['text_a'].to_list()
    labels = data['label'].to_list()
    data = [sentences, labels]
  else:
    data_path = args["test_data_path"]
    data = pd.read_csv(data_path, sep='\t', header=0) 
    sentences = data['text_a'].to_list()
    data = [sentences]
  return data

# dataset definition
class EmoDataset(Data.Dataset):
  def __init__(self, data:list, args:dict, with_labels:bool=True):
    model_name = args["model_name"]
    self.max_len = args["max_len"]
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.sentences = data[0]
    if with_labels:
      self.labels = data[1]
    self.with_labels = with_labels
  def __len__(self):
    return len(self.sentences)
  def __getitem__(self, index:int):
    sent = self.sentences[index]
    encoded_pair = self.tokenizer(
        sent,
        padding="max_length",
        truncation=True,
        max_length=self.max_len,
        return_tensors='pt'
    )
    token_ids = encoded_pair['input_ids'].squeeze(0)
    attn_masks = encoded_pair['attention_mask'].squeeze(0)
    token_type_ids = encoded_pair['token_type_ids'].squeeze(0)

    if self.with_labels:
      # trainset
      return token_ids, attn_masks, token_type_ids, self.labels[index]
    else:
      # testset
      return token_ids, attn_masks, token_type_ids
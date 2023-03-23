#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
# @Time  : 2023/03/21 19:29:05
# @Author: wd-2711
'''
from config import *

# model definition
class BertAnalysis(nn.Module):
    def __init__(self, args:dict):
      super(BertAnalysis, self).__init__()
      hidden_size, n_class, model_name = args["hidden_size"], args["n_class"], args["model_name"]
      self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
      self.linear = nn.Linear(hidden_size, n_class)
      self.dropout = nn.Dropout(0.5)
    
    def forward(self, X):
      input_ids, attn_masks, token_type_ids = X[0], X[1], X[2]
      with torch.no_gard():
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attn_masks,
            token_type_ids = token_type_ids
        )
      logits = self.linear(self.dropout(outputs.pooler_output))
      return logits
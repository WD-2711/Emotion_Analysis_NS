#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
# @Time  : 2023/03/21 19:58:06
# @Author: wd-2711
'''

from config import *

# TextCNN definition
class TextCNN(nn.Module):
  def __init__(self, args:dict):
    super(TextCNN, self).__init__()
    self.num_filters = args["num_filters"]
    self.filter_sizes = args["filter_sizes"]
    self.n_class = args["n_class"]
    self.hidden_size = args["hidden_size"]
    self.encode_layer = args["encode_layer"]
    self.num_filter_total = self.num_filters * len(self.filter_sizes)
    self.Weight = nn.Linear(self.num_filter_total, self.n_class, bias=False)
    self.bias = nn.Parameter(torch.ones([self.n_class]))
    self.filter_list = nn.ModuleList([nn.Conv2d(1, self.num_filters, kernel_size=(size, self.hidden_size)) for size in self.filter_sizes])
  def forward(self, x):
    # x=[batch_size, encode_layer, hidden_size]
    x = x.unsqueeze(1)
    # x=[batch_size, 1, encode_layer, hidden_size]
    pooled_outputs = []
    for i, conv in enumerate(self.filter_list):
      h = F.relu(conv(x))
      # h=[batch_size, num_filters, encode_layer-size+1, hidden_size-hidden_size+1]
      mp = nn.MaxPool2d(
          kernel_size=(self.encode_layer-self.filter_sizes[i]+1, 1)
      )
      pooled = mp(h).permute(0, 3, 2, 1)
      # pooled=[batch_size, 1, 1, num_filters]
      pooled_outputs.append(pooled)
    h_pool = torch.cat(pooled_outputs, len(self.filter_sizes))
    # h_pool=[batch_size, 1, 1, num_filters*len(self.filter_sizes)]
    h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])
    # h_pool_flat=[batch_size, num_filters*len(self.filter_sizes)]
    output = self.Weight(h_pool_flat) + self.bias
    # output=[batch_size, n_class]
    return output

# model definition
class BertBlendTextCNN(nn.Module):
    def __init__(self, args:dict):
      super(BertBlendTextCNN, self).__init__()
      hidden_size, n_class, model_name = args["hidden_size"], args["n_class"], args["model_name"]
      self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
      self.linear = nn.Linear(hidden_size, n_class)
      self.textCNN = TextCNN(args)
    
    def forward(self, X):
      input_ids, attn_masks, token_type_ids = X[0], X[1], X[2]
      # input_ids&attn_masks&token_type_ids=[batch_size, max_len]
      with torch.no_gard():
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attn_masks,
            token_type_ids = token_type_ids
        )
      hidden_states = outputs.hidden_states
      # hidden_states=13*[batch_size, max_len, hidden_size]
      cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)
      # cls_embeddings=[batch_size, 1, hidden_size]
      for i in range(2, 13):
        cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
      # cls_embeddings=[batch_size, 12, hidden_size]
      logits = self.textCNN(cls_embeddings)
      return logits
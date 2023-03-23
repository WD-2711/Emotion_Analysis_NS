import os
import time
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import jieba

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from transformers import AutoTokenizer, AutoModel, logging
import torch.nn.functional as F

# close warning
logging.set_verbosity_error()
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set args
args = {
        # test or train
        "mode": "test",
        # bert or bert_textcnn
        "model_type": "bert",
        # bert-base-chinese or chinese-bert-wwm
        "model_name": "hfl/chinese-bert-wwm",
        "data_path": './data/train_wash.tsv',
        "max_len": 100,
        "batch_size": 512,
        "hidden_size": 768,
        "n_class": 2,
        "epoches": 20,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "device": device,
        "fold_k": 10,
        "loss": 100000,
        "model_path": "./model_saved/best_model_",
        # bert+textcnn
        "encode_layer": 12,
        "filter_sizes": [2, 2, 2],
        "num_filters": 3,
        # test mode
        "test_model_dir": "./model_saved/",
        "test_data_path": "./data/test.tsv"
}
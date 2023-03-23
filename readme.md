# Emotion Analysis

&emsp; This demo is used to classify the `NLPCC14-SC.tsv` data set into positive speech and negative speech. This demo is doing sentiment analysis at the `sentence level`. There are two models, namely `bert-base` and `bert-TextCNN`.

&emsp; This demo first analyzes the `sentence length` and `word frequency` of the data, and draws a `box plot` to screen out sentences that are too long.  Then, manually fine-tune the labels of the data set to get `train_wash_manually.tsv`. Afterwards, using the `bert` Chinese pre-training model `chinese-bert-wwm`, two subsequent models can be selected for training. The first follow-up model is a simple fully connected layer, and the second is to splice each layer of the bert hidden layer and put it into textCNN for training. This demo also uses the `k-fold cross-validation` method, the default `k=10`.

&emsp;However, the result of model training is not very good, the accuracy rate is only `0.6652`. However, I believe this is not the optimal processing result of the model.

## Instructions for use

- Install related libraries.

```
pip3 install transformers, jieba
```

- `data_wash.py` is a data display module, in which word frequency display may require Chinese settings. The following is the Chinese setting configuration **(only for colab)**:

```
# 中文设置
!wget "https://www.wfonts.com/download/data/2014/06/01/simhei/simhei.zip"
!unzip "simhei.zip"
!rm "simhei.zip"
!mv SimHei.ttf /usr/share/fonts/truetype/

import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.font_manager import fontManager
 
fontManager.addfont('/usr/share/fonts/truetype/SimHei.ttf')
mpl.rc('font', family='SimHei')
```

- `config.py` is a parameter setting module, and the introduction of parameters will not be repeated here.

- run steps:

```
python3 data_wash.py
python3 main.py
```


# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:55:22 2017

@author: DART_HSU
"""


import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties

 #loading chinese font 
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

df = pd.read_csv("./input/creditcard.csv")

#Select only the anonymized features.
v_features = df.ix[:,1:29].columns
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)


for i, cn in enumerate(df[v_features]):
    fig,ax = plt.subplots()
    sns.distplot(df[cn][df.Class == 1], bins=50 , color="c")
    sns.distplot(df[cn][df.Class == 0], bins=50 , color="y")
    ax.set_xlabel('')
    ax.set_title('特徵的直方圖: ' + str(cn),fontproperties=font)
    fig.savefig('./image/'+str(cn)+'.png')

plt.show()
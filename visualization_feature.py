# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:55:22 2017

@author: DART_HSU
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

 #loading chinese font 
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

df = pd.read_csv("creditcard.csv")

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
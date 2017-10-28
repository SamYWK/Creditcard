# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:07:04 2017

@author: SamKao
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing

#load data
df = pd.read_csv('creditcard.csv')
normal_index = np.array(df[df.Class==0].index)
fraud_index = np.array(df[df.Class==1].index)
#under sampling
random_normal_index = np.random.choice(normal_index, len(fraud_index), replace = False)
under_sample_index = np.concatenate([random_normal_index, fraud_index])
under_sample_df = df.iloc[under_sample_index, :]

y = under_sample_df['Class']
X = under_sample_df.drop(['Class'], axis = 1)
#
#normalization
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
X['normalized_Amount'] = min_max_scaler.fit_transform(X['Amount'].values.reshape(-1,1))
X = X.drop(['Amount'], axis = 1)
print(X)


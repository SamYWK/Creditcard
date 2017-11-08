# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:07:04 2017

@author: SamKao
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

def load_data():
    df = pd.read_csv('creditcard.csv')
    normal_index = np.array(df[df.Class==0].index)
    fraud_index = np.array(df[df.Class==1].index)
    return df, normal_index, fraud_index


def under_sampling(df, normal_index, fraud_index):
    random_normal_index = np.random.choice(normal_index, len(fraud_index), replace = False)
    under_sample_index = np.concatenate([random_normal_index, fraud_index])
    under_sample_df = df.iloc[under_sample_index, :]
    
    y = under_sample_df['Class']
    X = under_sample_df.drop(['Class'], axis = 1)
    return X, y

def normalization_split_train_test(X, y):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X['normalized_Amount'] = min_max_scaler.fit_transform(X['Amount'].values.reshape(-1,1))
    X = X.drop(['Amount'], axis = 1)
    
    #split 80% for training , 20% for testing
    return train_test_split(X, y, train_size = 0.8, random_state = 0)

def logistic_regression(X_train, X_test, y_train):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    logreg_predict = logreg.predict(X_test)
    return logreg_predict
    


def main():
    df, normal_index, fraud_index = load_data()
    X, y = under_sampling(df, normal_index, fraud_index)
    X_train, X_test, y_train, y_test = normalization_split_train_test(X, y)
    predict = logistic_regression(X_train, X_test, y_train)
    print(recall_score(y_test.values, predict, average = 'binary'))
main()
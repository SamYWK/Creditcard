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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import sys

def select_file_name(input_number):
    file_name = "empty"
    if (input_number == '0'):
        sys.exit(0)
    elif (input_number == '1'):
        file_name = "creditcard.csv"
    elif (input_number == '2'):
        file_name = 'creditcard_feature_deleted.csv'
    elif (input_number == '3'):
        file_name = 'creditcard-simple.csv'
    else:
        print('Must select correct number !\n')
    return file_name

def oversampling(df, normal_index, fraud_index):
    #random select indices in fraud indices with replacement
    random_fraud_index = np.random.choice(fraud_index, len(normal_index), replace = True)
    over_sample_index = np.concatenate([normal_index, random_fraud_index])
    #shuffle the index
    np.random.shuffle(over_sample_index)
    over_sample_df = df.iloc[over_sample_index, :]
    
    y = over_sample_df['Class']
    X = over_sample_df.drop(['Class','Time'], axis = 1)
    return X.values, y.values

def normalization_train_test_split(X, y):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X['normalized_Amount'] = min_max_scaler.fit_transform(X['Amount'].values.reshape(-1,1))
    X = X.drop(['Amount'], axis = 1)
    
    #split 80% for training , 20% for testing
    return train_test_split(X, y, test_size = 0.2, random_state = 0)

def logistic_regression(X_train, X_test, y_train):
    logreg = LogisticRegression(C = 0.1)
    logreg.fit(X_train, y_train)
    logreg_predict = logreg.predict(X_test)
    logreg_score = logreg.decision_function(X_test)
    return logreg_predict, logreg_score
  
def main():
    #data selection
    while(True):
        print('Which file are you going to load?')
        print('\n1)creditcard.csv\
                 \n2)creditcard_feature_deleted.csv\
                 \n3)creditcard-simple.csv\
                 \n0)Exit')
        file_num = input('Input number : ')
        file_name = select_file_name(file_num)
        if file_name != 'empty':
            break
    df = pd.read_csv(file_name)

    #split
    X_train, X_test, y_train, y_test = normalization_train_test_split(df.drop(['Class','Time'], axis = 1), df['Class'])
    
    #oversampling training data
    train_normal_index = np.array(y_train[y_train[:] ==0].index)
    train_fraud_index = np.array(y_train[y_train[:] ==1].index)
    X_train_us, y_train_us = oversampling(df, train_normal_index, train_fraud_index)
    #oversampling testing data
    test_normal_index = np.array(y_test[y_test[:] ==0].index)
    test_fraud_index = np.array(y_test[y_test[:] ==1].index)
    X_test_us, y_test_us = oversampling(df, test_normal_index, test_fraud_index)
    
    #predict
    predict, score = logistic_regression(X_train.values, X_test.values, y_train.values)
    predict_us, score_us = logistic_regression(X_train_us, X_test_us, y_train_us)
    
    #Without undersampling
    TN, FP, FN, TP = confusion_matrix(y_test.values, predict).ravel()
    print('\n\nWithout oversampling')
    print('TN :', TN, 'FP :', FP, 'FN :', FN, 'TP :', TP)
    print('Recall score :', recall_score(y_test.values, predict, average = 'binary'))
    
    #With undersampling
    TN, FP, FN, TP = confusion_matrix(y_test_us, predict_us).ravel()
    print('\n\nWith oversampling')
    print('TN :', TN, 'FP :', FP, 'FN :', FN, 'TP :', TP)
    print('Recall score :', recall_score(y_test_us, predict_us, average = 'binary'))
main()
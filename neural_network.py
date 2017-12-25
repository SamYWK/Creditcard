# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:31:23 2017

@author: SamKao
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import time

def select_file_name(input_number):
    file_name = "empty"
    if (input_number == '0'):
        sys.exit(0)
    elif (input_number == '1'):
        file_name = "creditcard.csv"
    elif (input_number == '2'):
        file_name = 'creditcard_18_features.csv'
    elif (input_number == '3'):
        file_name = 'creditcard_24_features.csv'
    else:
        print('Must select correct number !\n')
    return file_name

def load_data_normalize_Amount(name):
    df = pd.read_csv(name)
    #normalize
    min_max_scaler = MinMaxScaler()
    df['normalized_Amount'] = min_max_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df = df.drop(['Amount'], axis = 1)
    return df

def oversampling(df, normal_index, fraud_index):
    random_fraud_index = np.random.choice(fraud_index, len(normal_index), replace = True)
    over_sample_index = np.concatenate([normal_index, random_fraud_index])
    #shuffle the index
    np.random.shuffle(over_sample_index)
    over_sample_df = df.iloc[over_sample_index, :]
    
    y = over_sample_df['Class']
    X = over_sample_df.drop(['Class','Time'], axis = 1)
    return X.values, y.values

def undersampling(df, normal_index, fraud_index):
    #random select indices in fraud indices with replacement
    random_normal_index = np.random.choice(normal_index, len(fraud_index), replace = False)
    under_sample_index = np.concatenate([random_normal_index, fraud_index])
    #shuffle the index
    np.random.shuffle(under_sample_index)
    under_sample_df = df.iloc[under_sample_index, :]
    
    y = under_sample_df['Class']
    X = under_sample_df.drop(['Class','Time'], axis = 1)
    return X.values, y.values

def KFold(X, y):
    #3 folds
    skf = StratifiedKFold(n_splits=3)
    
    threshold_range = np.arange(0.0, 1.0, 0.1)
    #store the best threshold
    best_threshold = -1
    best_score = -1
    #for every threshold
    for threshold in threshold_range:
        avg_score = 0
        for train_index, cross_index in skf.split(X, y):
            #train data
            y_predict = neural_network(X[train_index], y[train_index].reshape(-1, 1), X[cross_index], y[cross_index], threshold)
            #calculate f1_score
            score = f1_score(y[cross_index], y_predict, average = 'binary')
            avg_score += score
        #averaging the score
        avg_score = avg_score/3
        print(avg_score)
        #pick a best threshold
        if avg_score > best_score:
            best_score = avg_score
            best_threshold = threshold
    print(best_threshold)
    return best_threshold   

def neural_network(X_train, y_train, X_test, y_test, threshold):
    n, d = X_train.shape
    
    with tf.device('/gpu:0'):
        X_placeholder = tf.placeholder(tf.float32, [None, d])
        y_placeholder = tf.placeholder(tf.float32, [None, 1])
        l1, l1_Weights, l1_biases = add_layer(X_placeholder, d, 15, activation_function = tf.nn.sigmoid)
        prediction, pre_Weights, pre_biases  = add_layer(l1, 15, 1, activation_function = tf.nn.sigmoid)
    
        # the error between prediction and real data
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder - prediction), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        
        init = tf.global_variables_initializer()
        
        with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
            sess.run(init)
            #train neural network
            for i in range(1000):
                # training
                sess.run(train_step, feed_dict={X_placeholder: X_train, y_placeholder: y_train})
                #if i % 50 == 0:
                    # to see the step improvement
                    #print(sess.run(loss, feed_dict={X_placeholder: X_train, y_placeholder: y_train}))
            
            #test
            test_predict = tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(X_placeholder, l1_Weights) + l1_biases), pre_Weights) + pre_biases)
            test_predict = sess.run(test_predict, feed_dict={X_placeholder: X_test})
        
        for index in range(len(test_predict)):
            if test_predict[index] >= threshold:
                test_predict[index] = 1
            else:
                test_predict[index] = 0
    return test_predict


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs, Weights, biases

def main():
    #data selection
    while(True):
        print('Which file are you going to load?')
        print('\n1)creditcard.csv\
                 \n2)creditcard_18_features.csv\
                 \n3)creditcard_24_features.csv\
                 \n0)Exit')
        file_num = input('Input number : ')
        file_name = select_file_name(file_num)
        if file_name != 'empty':
            break
    #data preprocessing
    df = load_data_normalize_Amount(file_name)
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Time', 'Class'], axis = 1), df['Class'], test_size = 0.2, random_state = 0)
    
    #oversampling training data
    train_normal_index = np.array(y_train[y_train[:] ==0].index)
    train_fraud_index = np.array(y_train[y_train[:] ==1].index)
    X_train_os, y_train_os = oversampling(df, train_normal_index, train_fraud_index)
    #oversampling testing data
    test_normal_index = np.array(y_test[y_test[:] ==0].index)
    test_fraud_index = np.array(y_test[y_test[:] ==1].index)
    X_test_os, y_test_os = oversampling(df, test_normal_index, test_fraud_index)
    
    #cross-validation for unbalanced data
    threshold_unbalanced = 0.5 #KFold(X_train.values, y_train.values)  kFold works, but it takes a long time. You can try it! XD
    #undersampling data for cross-validation
    X_train_us, y_train_us = undersampling(df, train_normal_index, train_fraud_index)
    threshold_balanced = 0.6 #KFold(X_train_us, y_train_us)  You can try kFold
    
    #train neural network and return prediction
    print('Training unbalanced data and returning prediction...')
    start_time = time.time()
    predict = neural_network(X_train.values.astype(np.float32), y_train.values.reshape(-1, 1), X_test.values.astype(np.float32), y_test.values.reshape(-1, 1), threshold_unbalanced)
    end_time = time.time()
    unbalanced_time = end_time - start_time
    
    print('Training balanced data and returning prediction...')
    start_time = time.time()
    predict_os = neural_network(X_train_os.astype(np.float32), y_train_os.reshape(-1, 1), X_test_os.astype(np.float32), y_test_os.reshape(-1, 1), threshold_balanced)
    end_time = time.time()
    balanced_time = end_time - start_time
    
    #Without oversampling
    TN, FP, FN, TP = confusion_matrix(y_test.values, predict).ravel()
    print('\n\nWithout oversampling')
    print('TN :', TN, 'FP :', FP, 'FN :', FN, 'TP :', TP)
    print('Recall score :', recall_score(y_test.values, predict, average = 'binary'))
    print ("It cost %f sec" % unbalanced_time)
    
    #With oversampling
    TN, FP, FN, TP = confusion_matrix(y_test_os, predict_os).ravel()
    print('\n\nWith oversampling')
    print('TN :', TN, 'FP :', FP, 'FN :', FN, 'TP :', TP)
    print('Recall score :', recall_score(y_test_os, predict_os, average = 'binary'))
    print ("It cost %f sec" % balanced_time)
    
    return None

main()
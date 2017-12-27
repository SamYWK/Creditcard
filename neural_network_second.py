# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:22:13 2017

@author: DART_HSU
"""

import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import time
from matplotlib.font_manager import FontProperties

#function load_data :
#vesion that is different  function seleted, 0=original, 1=very similar, 2=less similar
#test_size that is splite rate, 0-1
#sampling that is simpling method, 0=none, 1=oversampling, 2=undersampling
def load_data (vesion=0, test_size=0.2, sampling=0):
    df = pd.read_csv("./input/creditcard.csv")
    
    #Drop all of the features that have very similar distributions between the two types of transactions.
    if vesion==1:     
        df = df.drop(['V28','V20','V15','V8'], axis =1)      
    elif vesion==2:
        df = df.drop(['V27','V25','V21','V15'], axis =1)
    elif vesion==3:
        df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
    
    #simpling method
    if sampling==1 :
        df = oversampling(df)
    elif sampling==2 :
        df = undersmapling(df)
    
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Time', 'Class'], axis = 1), df['Class'], test_size = 0.2, random_state = 0)
    
    #Names of all of the features in X_train.
    features = X_train.columns.values
    
    #Transform each feature in features so that it has a mean of 0 and standard deviation of 1; 
    #this helps with training the neural network.
    for feature in features:
        mean, std = df[feature].mean(), df[feature].std()
        X_train.loc[:, feature] = (X_train[feature] - mean) / std
        X_test.loc[:, feature] = (X_test[feature] - mean) / std

    return df, X_train ,X_test, y_train, y_test

def oversampling(df):
    Fraud = df[df.Class == 1]
    Normal = df[df.Class == 0]
    
    #Oversampling index 
    oversampling_fraud_index = np.random.choice(Fraud.index, len(Normal), replace = True )
    oversampling_df_index = np.append ( oversampling_fraud_index , Normal.index )
    
    oversampling_df = df.iloc[oversampling_df_index, :]
    
    return oversampling_df

def undersmapling(df):
    Fraud = df[df.Class == 1]
    Normal = df[df.Class == 0]

    #Undersampling index 
    undersampling_normal_index = np.random.choice(Normal.index, len(Fraud), replace = False )
    undersampling_index = np.append ( undersampling_normal_index , Fraud.index )
    
    undersampling_df = df.iloc[undersampling_index, :]

    return undersampling_df


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs, Weights, biases


def neural_network(X_train, y_train, X_test, y_test, threshold):
    learning_rate = 0.1
    n, d = X_train.shape
    cost_summary = [] # Record cost values for plot
    
    with tf.device('/gpu:0'):
        X_placeholder = tf.placeholder(tf.float32, [None, d])
        y_placeholder = tf.placeholder(tf.float32, [None, 1])
        l1, l1_Weights, l1_biases = add_layer(X_placeholder, d, 35, activation_function = tf.nn.sigmoid)
        prediction, pre_Weights, pre_biases  = add_layer(l1, 35, 1, activation_function = None)
    
        # the error between prediction and real data
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_placeholder, logits = prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        
        init = tf.global_variables_initializer()
        
        with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
            sess.run(init)
            #train neural network
            for i in range(1000):
                # training
                sess.run(train_step, feed_dict={X_placeholder: X_train, y_placeholder: y_train})
                if i % 50 == 0:
                    # to see the step improvement
                    cost_summary.append( sess.run(cross_entropy, feed_dict={X_placeholder: X_train, y_placeholder: y_train}))
            #test
            test_predict = tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(X_placeholder, l1_Weights) + l1_biases), pre_Weights) + pre_biases)
            test_predict = sess.run(test_predict, feed_dict={X_placeholder: X_test})
        
        for index in range(len(test_predict)):
            if test_predict[index] >= threshold:
                test_predict[index] = 1
            else:
                test_predict[index] = 0
                
    return test_predict, cost_summary

def show_result(title, summary, y, y_, cost_time):
    #loading chinese font 
    font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
    info = ''
    
    TN, FP, FN, TP = confusion_matrix(y, y_).ravel()
    info += 'TN :'+ str(TN)+ ' FP :'+ str(FP) + '\n'
    info += 'FN :'+ str(FN) + ' TP :'+ str(TP) + '\n'
    info += 'Recall score : %.2f\n' %recall_score(y_test.values, predict, average = 'binary')
    info += 'It cost %.2f sec' %cost_time

    f, ax = plt.subplots(1, 1, sharex=True, figsize=(8,4))
    ax.plot(summary) # blue
    ax.set_title(title, fontsize=20)
    ax.text(10, 0.5, str(info), fontsize=14)
    plt.xlabel('次數',fontproperties=font)
    plt.ylabel('Cost值',fontproperties=font)
    plt.show()
        
    return 0

if __name__ == "__main__":
    df,X_train, X_test, y_train, y_test =  load_data(0, 0.2, 0)

    #train neural network and return prediction
    print('Training unbalanced data and returning prediction...')
    
    df, X_train_us, y_train_us, X_test_us, y_test_us = load_data(0, 0.2, 2)
    #cross-validation for unbalanced data
    #KFold(X_train.values, y_train.values)  kFold works, but it takes a long time. You can try it! XD
    threshold_unbalanced = 0.5 
    start_time = time.time()
    predict, cost_summary = neural_network(X_train.values.astype(np.float32), y_train.values.reshape(-1, 1), X_test.values.astype(np.float32), y_test.values.reshape(-1, 1), threshold_unbalanced)
    end_time = time.time()
    unbalanced_time = end_time - start_time
    
    print('Training balanced data and returning prediction...')
    #data preprocessing ,oversampling  data
    df, X_train_os, X_test_os, y_train_os, y_test_os = load_data(0, 0.2, 1)
    threshold_balanced = 0.6 #KFold(X_train_us, y_train_us)  You can try kFold
    start_time = time.time()
    predict_os, cost_summary_os = neural_network(X_train_os.values.astype(np.float32), y_train_os.values.reshape(-1, 1), X_test_os.values.astype(np.float32), y_test_os.values.reshape(-1, 1), threshold_balanced)
    end_time = time.time()
    balanced_time_os = end_time - start_time
    
    print('Training balanced data and returning prediction (vesion 1)...')
    #data preprocessing ,oversampling  data
    df, X_train_os, X_test_os, y_train_os, y_test_os1 = load_data(1, 0.2, 1)
    threshold_balanced = 0.6 #KFold(X_train_us, y_train_us)  You can try kFold
    start_time = time.time()
    predict_os1, cost_summary_os1 = neural_network(X_train_os.values.astype(np.float32), y_train_os.values.reshape(-1, 1), X_test_os.values.astype(np.float32), y_test_os.values.reshape(-1, 1), threshold_balanced)
    end_time = time.time()
    balanced_time_os1 = end_time - start_time

    print('Training balanced data and returning prediction (vesion 2)...')
    #data preprocessing ,oversampling  data
    df, X_train_os, X_test_os, y_train_os, y_test_os2 = load_data(2, 0.2, 1)
    threshold_balanced = 0.6 #KFold(X_train_us, y_train_us)  You can try kFold
    start_time = time.time()
    predict_os2, cost_summary_os2 = neural_network(X_train_os.values.astype(np.float32), y_train_os.values.reshape(-1, 1), X_test_os.values.astype(np.float32), y_test_os.values.reshape(-1, 1), threshold_balanced)
    end_time = time.time()
    balanced_time_os2 = end_time - start_time
    
    print('Training balanced data and returning prediction (vesion 3)...')
    #data preprocessing ,oversampling  data
    df, X_train_os, X_test_os, y_train_os, y_test_os3 = load_data(3, 0.2, 1)
    threshold_balanced = 0.6 #KFold(X_train_us, y_train_us)  You can try kFold
    start_time = time.time()
    predict_os3, cost_summary_os3 = neural_network(X_train_os.values.astype(np.float32), y_train_os.values.reshape(-1, 1), X_test_os.values.astype(np.float32), y_test_os.values.reshape(-1, 1), threshold_balanced)
    end_time = time.time()
    balanced_time_os3 = end_time - start_time        
    
    print('showing...')   
    show_result('unbalanced data', cost_summary, y_test, predict, unbalanced_time)
    show_result('balanced data(vesion 0)', cost_summary_os, y_test_os, predict_os, balanced_time_os)
    show_result('balanced data(vesion 1)', cost_summary_os1, y_test_os1, predict_os1, balanced_time_os1)
    show_result('balanced data(vesion 2)', cost_summary_os2, y_test_os2, predict_os2, balanced_time_os2)
    show_result('balanced data(vesion 3)', cost_summary_os3, y_test_os3, predict_os3, balanced_time_os3)
    
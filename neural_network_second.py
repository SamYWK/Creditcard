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
from sklearn.linear_model import LogisticRegression

#function load_data :
#version that is different  function seleted, 0=original, 1=very similar, 2=less similar
#test_size that is splite rate, 0-1
#sampling that is simpling method, 0=none, 1=oversampling, 2=undersampling
def load_data (version=0, test_size=0.2, sampling=0):
    df = pd.read_csv("./input/creditcard.csv")
    
    #Drop all of the features that have very similar distributions between the two types of transactions.
    if version==1:   #creditcard_24_features
        df = df.drop(['V28','V20','V15','V8'], axis =1)      
    elif version==2:
        df = df.drop(['V27','V25','V21','V15'], axis =1)
    elif version==3: #creditcard_18_features
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
    learning_rate = 0.105
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
                if i % 10 == 0:
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


def logistic_regression(X_train, X_test, y_train, c):
    logreg = LogisticRegression(C = c)
    
    logreg.fit(X_train, y_train)
    logreg_predict = logreg.predict(X_test)
    logreg_score = logreg.decision_function(X_test)
    
    return logreg_predict, logreg_score

def show_result(title, summary, y, y_, cost_time):
    #loading chinese font 
    font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
    info = ''
    
    TN, FP, FN, TP = confusion_matrix(y, y_).ravel()
    info += 'TN : '+ str(TN)+ ' FP : '+ str(FP) + '\n'
    info += 'FN : '+ str(FN) + ' TP : '+ str(TP) + '\n'
    info += 'Recall score : %.3f\n' %recall_score(y, y_, average = 'binary')
    info += 'It cost %.2f sec' %cost_time

    f, ax = plt.subplots(1, 1, sharex=True, figsize=(8,4))
    plt.xlim(0,100)
    plt.ylim(0,2)
    ax.plot(summary) # blue
    ax.set_title(title, fontsize=20)
    ax.text(50, 1, str(info), fontsize=14)
    plt.xlabel('遞迴次數(每10次)',fontproperties=font)
    plt.ylabel('Cross_entropy',fontproperties=font)
    plt.show()
    f.savefig('./image/'+str(title)+'.png')
        
    return 0

def show_cross_comparison_result(title, summarys, cost_times):
    #loading chinese font 
    font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
    info = ''
    
    print (summarys)
    print (cost_times)
    
    for indx in range(0, 3):
        info += 'version'+str(indx)+' : %.2f sec \n'%cost_times[indx]

    
    f, ax = plt.subplots(1, 1, sharex=True, figsize=(8,4))
    plt.xlim(0,100)
    plt.ylim(0,2)
    
    line1, = ax.plot(summarys[0]) # blue
    line2, = ax.plot(summarys[1]) # blue
    line3, = ax.plot(summarys[2]) # blue
        
    ax.set_title(title, fontsize=20)
    ax.text(50, 1, str(info), fontsize=14)
    plt.xlabel('遞迴次數(每10次)',fontproperties=font)
    plt.ylabel('Cross_entropy',fontproperties=font)
    
    ax.legend((line1, line2, line3),('version0', 'version1' ,'version2'))
    plt.show()
    f.savefig('./image/'+str(title)+'.png')
        
    return 0

def show_all_recallscore_result_bar(logistic_scores, NN_scores):
    N = 5
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars
    
    f, ax = plt.subplots(figsize=(10,6))
    
    rects1 = ax.bar(ind, list(logistic_scores), width, color='r')
    rects2 = ax.bar(ind + width, list (NN_scores), width, color='y')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1*height,
                    '%.3f' % height,
                    ha='center', va='bottom')
        
    autolabel(rects1)
    autolabel(rects2)
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores(%)')
    ax.set_title('Recall Scores')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Unblanced Data', 'version0', 'version1', 'version2','version3'))
    ax.legend((rects1[0], rects2[0]), ('Logistic', 'Neural Network'))
    
    f.savefig ('./image/RecallScore比較圖.png')  
 
    plt.show()
    

def show_all_cost_time_result_bar(logistic_times, NN_times):
    N = 5
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars
    
    f, ax = plt.subplots(figsize=(10,6))
    
    rects1 = ax.bar(ind, list(logistic_times), width, color='r')
    rects2 = ax.bar(ind + width, list (NN_times), width, color='y')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1*height,
                    '%.2f' % height,
                    ha='center', va='bottom')
        
    autolabel(rects1)
    autolabel(rects2)
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Time(s)')
    ax.set_title('Cost Time')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Unblanced Data', 'version0', 'version1', 'version2','version3'))
    ax.legend((rects1[0], rects2[0]), ('Logistic', 'Neural Network'))
    
    f.savefig ('./image/CostTime比較圖.png')  
 
    plt.show()

if __name__ == "__main__":
    logistic_scores = []
    logistic_times = []
    NN_scores = []
    NN_times = []
    NN_cross_lines = []
    NN_cost_times = []
    

    print('Training NN...')
    
    #train neural network and return prediction
    print('Training unbalanced data and returning prediction...')
    df, X_train, X_test, y_train, y_test = load_data(0, 0.2, 0)
    threshold_unbalanced = 0.5 
    
    start_time = time.time()
    predict, cost_summary = neural_network(X_train.values.astype(np.float32), y_train.values.reshape(-1, 1), X_test.values.astype(np.float32), y_test.values.reshape(-1, 1), threshold_unbalanced)
    end_time = time.time()
    unbalanced_time = end_time - start_time
    NN_scores.append (recall_score(y_test, predict, average = 'binary'))
    NN_times.append( unbalanced_time )
    
    print('Training balanced data and returning prediction...')
    #data preprocessing ,oversampling  data
    df, X_train_os, X_test_os, y_train_os, y_test_os = load_data(0, 0.2, 1)
    threshold_balanced = 0.6 #KFold(X_train_us, y_train_us)  You can try kFold
    
    start_time = time.time()
    predict_os, cost_summary_os = neural_network(X_train_os.values.astype(np.float32), y_train_os.values.reshape(-1, 1), X_test_os.values.astype(np.float32), y_test_os.values.reshape(-1, 1), threshold_balanced)
    end_time = time.time()
    balanced_time_os = end_time - start_time
    NN_scores.append (recall_score(y_test_os, predict_os, average = 'binary'))
    NN_times.append( balanced_time_os )
    
    print('Training balanced data and returning prediction (version 1)...')
    #data preprocessing ,oversampling  data
    df, X_train_os, X_test_os, y_train_os, y_test_os1 = load_data(1, 0.2, 1)
    threshold_balanced = 0.6 #KFold(X_train_us, y_train_us)  You can try kFold
    
    start_time = time.time()
    predict_os1, cost_summary_os1 = neural_network(X_train_os.values.astype(np.float32), y_train_os.values.reshape(-1, 1), X_test_os.values.astype(np.float32), y_test_os.values.reshape(-1, 1), threshold_balanced)
    end_time = time.time()
    balanced_time_os1 = end_time - start_time
    NN_scores.append (recall_score(y_test_os1, predict_os1, average = 'binary'))
    NN_times.append( balanced_time_os1 )

#    print('Training balanced data and returning prediction (version 2)...')
#    #data preprocessing ,oversampling  data
#    df, X_train_os, X_test_os, y_train_os, y_test_os2 = load_data(2, 0.2, 1)
#    threshold_balanced = 0.6 #KFold(X_train_us, y_train_us)  You can try kFold
#    start_time = time.time()
#    predict_os2, cost_summary_os2 = neural_network(X_train_os.values.astype(np.float32), y_train_os.values.reshape(-1, 1), X_test_os.values.astype(np.float32), y_test_os.values.reshape(-1, 1), threshold_balanced)
#    end_time = time.time()
#    balanced_time_os2 = end_time - start_time
    
    print('Training balanced data and returning prediction (version 3)...')
    #data preprocessing ,oversampling  data
    df, X_train_os, X_test_os, y_train_os, y_test_os3 = load_data(3, 0.2, 1)
    threshold_balanced = 0.6 #KFold(X_train_us, y_train_us)  You can try kFold
    
    start_time = time.time()
    predict_os3, cost_summary_os3 = neural_network(X_train_os.values.astype(np.float32), y_train_os.values.reshape(-1, 1), X_test_os.values.astype(np.float32), y_test_os.values.reshape(-1, 1), threshold_balanced)
    end_time = time.time()
    balanced_time_os3 = end_time - start_time
    NN_scores.append (recall_score(y_test_os3, predict_os3, average = 'binary'))
    NN_times.append( balanced_time_os3 )
    
    print('Training balanced data and returning prediction (version 3, 0.2, undersmapling)...')
    #data preprocessing ,oversampling  data
    df, X_train_os, X_test_os, y_train_os, y_test_us3 = load_data(3, 0.2, 2)
    threshold_balanced = 0.6 #KFold(X_train_us, y_train_us)  You can try kFold
    
    start_time = time.time()
    predict_us3, cost_summary_us3 = neural_network(X_train_os.values.astype(np.float32), y_train_os.values.reshape(-1, 1), X_test_os.values.astype(np.float32), y_test_os.values.reshape(-1, 1), threshold_balanced)
    end_time = time.time()
    balanced_time_us3 = end_time - start_time
    NN_scores.append (recall_score(y_test_us3, predict_us3, average = 'binary'))   
    NN_times.append( balanced_time_us3 )         
    
#    print('Showing...')   
    show_result('Unbalanced Data', cost_summary, y_test, predict, unbalanced_time)
    show_result('version 0', cost_summary_os, y_test_os, predict_os, balanced_time_os)
    show_result('version 1', cost_summary_os1, y_test_os1, predict_os1, balanced_time_os1)
#    show_result('Balanced Data(version 2)', cost_summary_os2, y_test_os2, predict_os2, balanced_time_os2)
    show_result('version 2', cost_summary_os3, y_test_os3, predict_os3, balanced_time_os3)
    show_result('version 3', cost_summary_us3, y_test_us3, predict_us3, balanced_time_us3)
    
    NN_cross_lines.append (cost_summary_os )
    NN_cross_lines.append (cost_summary_os1 )    
    NN_cross_lines.append (cost_summary_os3 )    
    
    NN_cost_times.append (balanced_time_os )
    NN_cost_times.append (balanced_time_os1 )
    NN_cost_times.append (balanced_time_os3 )    
#    
    show_cross_comparison_result ('Compare Three Version\'s Cost', NN_cross_lines, NN_cost_times)
#    
    print('Training losgitic...')

    print('Training unbalanced data and returning prediction...')
    df, X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = load_data(0, 0.2, 0)
    
    start_time = time.time()
    predict_logistic, score_logistic = logistic_regression(X_train_logistic, X_test_logistic, y_train_logistic, 0.5)
    end_time = time.time()
    cost_time = end_time - start_time
    
    logistic_scores.append (recall_score(y_test_logistic, predict_logistic, average = 'binary'))
    logistic_times.append (cost_time)         
    
    print('Traing losgitic (version 0)')    
    df, X_train_logistic_os, X_test_logistic_os, y_train_logistic_os, y_test_logistic_os = load_data(1, 0.2, 1)
    
    start_time = time.time()
    predict_logistic_os, score_logistic_os = logistic_regression(X_train_logistic_os, X_test_logistic_os, y_train_logistic_os, 0.5) 
    end_time = time.time()
    cost_time = end_time - start_time
    
    logistic_scores.append (recall_score(y_test_logistic_os, predict_logistic_os, average = 'binary'))
    logistic_times.append (cost_time)     
    
    print('Traing losgitic (version 1)')    
    df, X_train_logistic_os1, X_test_logistic_os1, y_train_logistic_os1, y_test_logistic_os1 = load_data(1, 0.2, 1)
    
    start_time = time.time()    
    predict_logistic_os1, score_logistic_os1 = logistic_regression(X_train_logistic_os1, X_test_logistic_os1, y_train_logistic_os1, 0.5) 
    end_time = time.time()
    cost_time = end_time - start_time 
    
    logistic_scores.append (recall_score(y_test_logistic_os1, predict_logistic_os1, average = 'binary'))
    logistic_times.append (cost_time)     
    
    print('Traing losgitic (version 3)')    
    df, X_train_logistic_os3, X_test_logistic_os3, y_train_logistic_os3, y_test_logistic_os3 = load_data(3, 0.2, 1)

    start_time = time.time()    
    predict_logistic_os3, score_logistic_os3 = logistic_regression(X_train_logistic_os3, X_test_logistic_os3, y_train_logistic_os3, 0.5) 
    end_time = time.time()
    cost_time = end_time - start_time
    
    logistic_scores.append (recall_score(y_test_logistic_os3, predict_logistic_os3, average = 'binary'))  
    logistic_times.append (cost_time)         
    
    print('Traing losgitic (version 3, 0.2, undersmapling)')    
    df, X_train_logistic_us3, X_test_logistic_us3, y_train_logistic_us3, y_test_logistic_us3 = load_data(3, 0.2, 2)
 
    start_time = time.time()
    predict_logistic_us3, score_logistic_us3 = logistic_regression(X_train_logistic_us3, X_test_logistic_us3, y_train_logistic_us3, 0.5)
    end_time = time.time()
    cost_time = end_time - start_time
    
    logistic_scores.append (recall_score(y_test_logistic_us3, predict_logistic_us3, average = 'binary'))  
    logistic_times.append (cost_time)           
    
    show_all_recallscore_result_bar(logistic_scores, NN_scores)
    show_all_cost_time_result_bar(logistic_times, NN_times)
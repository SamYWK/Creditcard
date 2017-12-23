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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def oversampling(df, normal_index, fraud_index):
    random_fraud_index = np.random.choice(fraud_index, len(normal_index), replace = True)
    over_sample_index = np.concatenate([normal_index, random_fraud_index])
    over_sample_df = df.iloc[over_sample_index, :]
    
    y = over_sample_df['Class']
    X = over_sample_df.drop(['Class','Time'], axis = 1)
    return X, y

def normalization_train_test_split(X, y):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X['normalized_Amount'] = min_max_scaler.fit_transform(X['Amount'].values.reshape(-1,1))
    X = X.drop(['Amount'], axis = 1)
    
    #split 80% for training , 20% for testing
    return train_test_split(X, y, train_size = 0.8, random_state = 0)

def logistic_regression(X_train, X_test, y_train):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    logreg_predict = logreg.predict(X_test)
    logreg_score = logreg.decision_function(X_test)
    return logreg_predict, logreg_score

def pr_curve(y_test, y_score, figure_num):
    #average precision
    average_precision = average_precision_score(y_test, y_score)
    
    plt.figure(figure_num)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.3f}'.format(average_precision))
    plt.show()
    return None

def printing_Kfold_scores(X, y, oversampling = False):
    
    
    fold = KFold(n_splits = 5,shuffle=False)
    
    # Different C parameters
    c_param_range = [0.01,0.1,1,10,100]

    results_table = pd.DataFrame(index = range(len(c_param_range)), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range
    
    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        for train_index, cross_index in fold.split(X):
            if(oversampling):
                train_normal_index = np.array(y[y[train_index] == 0].index)
                train_fraud_index = np.array(y[y == 1].index)
                
            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C = c_param, penalty = 'l2')

            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(X.iloc[train_index, :], y.iloc[train_index])

            # Predict values using the test indices in the training data
            y_pred = lr.predict(X.iloc[cross_index,:])

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y.iloc[cross_index], y_pred)
            recall_accs.append(recall_acc)
            print('recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.loc[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    best_c = results_table[results_table['Mean recall score'].idxmax()]['C_parameter']
    
    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c
  
def main():
    df = pd.read_csv('creditcard.csv')

    #split
    X_train, X_test, y_train, y_test = normalization_train_test_split(df.drop(['Class','Time'], axis = 1), df['Class'])
    
    #corss validation
    #non_oversampling_c = printing_Kfold_scores(X_train.values, y_train.values)
    oversampling_c = printing_Kfold_scores(X_train, y_train, oversampling = True)
    
    '''
    #predict
    predict, score = logistic_regression(X_train, X_test, y_train)
    predict_us, score_us = logistic_regression(X_train_us, X_test_us, y_train_us)
    
    #Without undersampling
    TN, FP, FN, TP = confusion_matrix(y_test.values, predict).ravel()
    print('\n\nWithout oversampling')
    print('TN :', TN, 'FP :', FP, 'FN :', FN, 'TP :', TP)
    print('Recall score :', recall_score(y_test, predict, average = 'binary'))
    #pr_curve(y_test, score, 1)
    
    #With undersampling
    TN, FP, FN, TP = confusion_matrix(y_test_us.values, predict_us).ravel()
    print('\n\nWith oversampling')
    print('TN :', TN, 'FP :', FP, 'FN :', FN, 'TP :', TP)
    print('Recall score :', recall_score(y_test_us, predict_us, average = 'binary'))
    #pr_curve(y_test_us, score_us, 2)'''
main()
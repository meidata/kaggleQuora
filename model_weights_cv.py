# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:17:12 2017

@author: n000153994
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:08:27 2017

@author: meiyi
"""

import pandas as pd
import numpy as np
from scipy import sparse as ssp


path_feature = 'D:\\kaggle_quora\\features\\'

# basic features ---- features engineering

test_data = pd.DataFrame()

for i in range(0,10):
    filename = 'test_'+str(i)+'_quora_features.pkl'
    data = pd.read_pickle(path_feature+filename)
    test_data = test_data.append(data)
    
    
train_data = pd.read_pickle(path_feature + 'train_quora_features.pkl')


#train_w2v_q1 = np.load(path_feature+'train_q1_w2v_google.pkl')
#train_w2v_q1 = pd.DataFrame(train_w2v_q1,columns=['q1_' + i for i in list(map(str,range(0,train_w2v_q1.shape[1])))])
#  
#train_w2v_q2 = np.load(path_feature+'train_q2_w2v_google.pkl')
#train_w2v_q2 = pd.DataFrame(train_w2v_q2,columns=['q2_' + i for i in list(map(str,range(0,train_w2v_q2.shape[1])))])




# magic features 

train_comb = pd.read_pickle(path_feature+'magic_feature_train.pkl')
test_comb = pd.read_pickle(path_feature+'magic_feature_test.pkl')


# features stacking
 

train_data['weights']= [ np.random.uniform(0.2,0.25) if x == 1 else
                         np.random.uniform(0.6,0.65) for x in train_data['is_duplicate']]


train_features = pd.concat([train_data[train_data.columns.difference(['question1', 'question2'])],
                             train_comb[train_comb.columns.difference(['id','is_duplicate'])]], axis=1)
    #.tocsr()
    

test_features = pd.concat([test_data[test_data.columns.difference(['question1', 'question2'])],
                            test_comb[test_comb.columns.difference(['id'])]],axis=1)
    #.tocsr()
    


from sklearn.model_selection import train_test_split

feature_col = train_features.columns.difference(['id']).values.tolist()

pos_train, pos_test = train_test_split(train_features.ix[train_features['is_duplicate']==1,feature_col], test_size = 0.3)
neg_train, neg_test = train_test_split(train_features.ix[train_features['is_duplicate']==0,feature_col], test_size = 0.3)

train_X = pos_train.append(neg_train)
test_X = pos_test.append(neg_test)

train_y = train_X.is_duplicate.values
weight_X = train_X.weights.values
train_X = train_X[train_X.columns.difference(['is_duplicate','weights'])]

test_y = test_X.is_duplicate.values
weight_x = test_X.weights.values
test_X = test_X[test_X.columns.difference(['is_duplicate','weights'])]

# Set our parameters for xgboost

import xgboost as xgb

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 10

d_train = xgb.DMatrix(train_X, label=train_y,weight=weight_X)
d_valid = xgb.DMatrix(test_X, label=test_y,weight=weight_x) 

watchlist = [(d_train, 'train'), (d_valid, 'valid')]


bst = xgb.train(params, d_train, 1500, watchlist, early_stopping_rounds=5, verbose_eval=10)
               # , feval = kappa)

feature_col_test = train_X.columns.values.tolist()

d_test = xgb.DMatrix(test_features.ix[:,feature_col_test])
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = test_comb['id']
sub['is_duplicate'] = p_test
path = 'C:\\Users\\N000153994\\Desktop\\kaggle\\'
sub.to_csv(path+'xgb_1705_0.2_0.6_9.csv', index=False)
    



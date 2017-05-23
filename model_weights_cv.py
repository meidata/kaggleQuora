#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:08:27 2017

@author: meiyi
"""

import platform
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def setPath():
    if platform.system() == 'Darwin':
        path_w2v = '/Volumes/MyPassport/kaggle_quora/w2v_pretrained/'
        path_data= '/Volumes/MyPassport/kaggle_quora/data/'
        path_feature = '/Volumes/MyPassport/kaggle_quora/features/'
 
        return path_w2v,path_data,path_feature 
    elif platform.system() == 'Windows':
        path_w2v = 'D:\\kaggle_quora\\w2v_pretrained\\'
        path_data= 'D:\\kaggle_quora\\data\\'
        path_feature = 'D:\\kaggle_quora\\features\\'
        return path_w2v,path_data,path_feature 
        
path_w2v,path_data,path_feature  = setPath()


# basic features ---- features engineering

test_data = pd.DataFrame()

for i in range(0,10):
    filename = 'test_'+str(i)+'_quora_features.pkl'
    data = pd.read_pickle(path_feature+filename)
    test_data = test_data.append(data)
    
    
train_data = pd.read_pickle(path_feature + 'train_quora_features.pkl')

'''
ques = pd.concat([train_data[['question1', 'question2']], \
        test_data[['question1', 'question2']]], axis=0).reset_index(drop='index')


q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])
        
        
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))



train_data['q1_q2_intersect'] = train_data.apply(q1_q2_intersect, axis=1, raw=True)
test_data['q1_q2_intersect'] = test_data.apply(q1_q2_intersect, axis=1, raw=True)


train_data[['q1_q2_intersect']].to_pickle(path_feature + 'train_intersect.pkl')
test_data[['q1_q2_intersect']].to_pickle(path_feature + 'test_intersect.pkl')
'''


#train_w2v_q1 = np.load(path_feature+'train_q1_w2v_google.pkl')
#train_w2v_q1 = pd.DataFrame(train_w2v_q1,columns=['q1_' + i for i in list(map(str,range(0,train_w2v_q1.shape[1])))])
#  
#train_w2v_q2 = np.load(path_feature+'train_q2_w2v_google.pkl')
#train_w2v_q2 = pd.DataFrame(train_w2v_q2,columns=['q2_' + i for i in list(map(str,range(0,train_w2v_q2.shape[1])))])

train_porter_intersec = pd.DataFrame(pd.read_pickle(path_feature+'train_porter_interaction.pkl'),
                                     columns = ['porter_intersec'])
test_porter_intersec = pd.DataFrame(pd.read_pickle(path_feature+'test_porter_interaction.pkl'),
                                     columns = ['porter_intersec'])



train_intersec = pd.read_pickle(path_feature + 'train_intersect.pkl')
test_intersec = pd.read_pickle(path_feature + 'test_intersect.pkl')


# magic features 

train_comb = pd.read_pickle(path_feature+'magic_feature_train.pkl')
test_comb = pd.read_pickle(path_feature+'magic_feature_test.pkl')


# features stacking
 

train_data['weights']= [ np.random.uniform(0.2,0.21) if x == 1 else
                         np.random.uniform(0.8,0.81) for x in train_data['is_duplicate']]


train_features = pd.concat([train_data[train_data.columns.difference(['question1', 'question2'])],
                                       train_porter_intersec,
                                       train_intersec,
                             train_comb[train_comb.columns.difference(['id','is_duplicate','q1_hash', 'q2_hash'])]], axis=1)
    #.tocsr()
    

test_features = pd.concat([test_data[test_data.columns.difference(['question1', 'question2'])],
                                     test_porter_intersec,
                                     test_intersec,
                            test_comb[test_comb.columns.difference(['q1_hash', 'q2_hash','id'])]],axis=1)
    #.tocsr()

'''
np.save(path_feature +'train_feature_with_intersec_no_hash.npy',
        train_features[train_features.columns.difference(['id','q1_hash','q2_hash','is_duplicate'])].values,allow_pickle=True)

np.save(path_feature +'test_feature_with_intersec_no_hash.npy',
        test_features[test_features.columns.difference(['id','q1_hash','q2_hash'])].values,allow_pickle=True)

np.save(path_feature +'train_is_duplicate.npy',
        train_features['is_duplicate'].values,allow_pickle=True)
'''

from sklearn.model_selection import train_test_split

feature_col = train_features.columns.difference(['id']).values.tolist()

pos_train, pos_test = train_test_split(train_features.ix[train_features['is_duplicate']==1,feature_col], test_size = 0.3)
neg_train, neg_test = train_test_split(train_features.ix[train_features['is_duplicate']==0,feature_col], test_size = 0.3)

train_X = pos_train.append(neg_train)
test_X = pos_test.append(neg_test)

train_y = train_X.is_duplicate.values
weight_X = train_X.weights.values
train_X = train_X[train_X.columns.difference(['is_duplicate'])] # remove weights for ensemble

test_y = test_X.is_duplicate.values
weight_x = test_X.weights.values
test_X = test_X[test_X.columns.difference(['is_duplicate'])]

# Set our parameters for xgboost

import xgboost as xgb

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 9

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

sub.to_csv(path_data+'xgb_2005_0.2_0.6_9_no_intersec_onlyporter_.csv', index=False)



    



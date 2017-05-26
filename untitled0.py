#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 23:09:33 2017

@author: meiyi
"""

from scipy.sparse import hstack,coo_matrix
from sklearn.preprocessing import MinMaxScaler


train_w2v_q1 = np.load(path_feature+'train_q1_w2v_google.pkl')
train_w2v_q1 = pd.DataFrame(train_w2v_q1,columns=['q1_' + i for i in list(map(str,range(0,train_w2v_q1.shape[1])))])

train_w2v_q2 = np.load(path_feature+'train_q2_w2v_google.pkl')
train_w2v_q2 = pd.DataFrame(train_w2v_q2,columns=['q2_' + i for i in list(map(str,range(0,train_w2v_q2.shape[1])))])


test_w2v_q1 = pd.DataFrame()

for i in range(0,10):
    filename = 'test_'+str(i)+'_q1_w2v_google.pkl'
    ary = np.load(path_feature+filename)
    data = pd.DataFrame(ary,columns=['q1_' + i for i in list(map(str,range(0,ary.shape[1])))])
    test_w2v_q1 = test_w2v_q1.append(data)
    
test_w2v_q1 = test_w2v_q1.as_matrix()

test_w2v_q2 = pd.DataFrame()

for i in range(0,10):
    filename = 'test_'+str(i)+'_q2_w2v_google.pkl'
    ary = np.load(path_feature+filename)
    data = pd.DataFrame(ary,columns=['q2_' + i for i in list(map(str,range(0,ary.shape[1])))])
    test_w2v_q2 = test_w2v_q2.append(data)
    
test_w2v_q2 = test_w2v_q2.as_matrix()

train_df = pd.concat([train_comb[['is_duplicate']],train_w2v_q1,train_w2v_q1],axis = 1)


from sklearn.model_selection import train_test_split



pos_train, pos_test = train_test_split(train_df.ix[train_df['is_duplicate']==1,], test_size = 0.3)
neg_train, neg_test = train_test_split(train_df.ix[train_df['is_duplicate']==0,], test_size = 0.3)


train_X = pos_train.append(neg_train)
test_X = pos_test.append(neg_test)


feature_col = train_df.columns.difference(['is_duplicate']).values.tolist()


scaler = MinMaxScaler()
scaler.fit(np.vstack([np.nan_to_num(train_X[feature_col].as_matrix()),
                                    np.nan_to_num(test_X[feature_col].as_matrix())]))

train_y = train_X.is_duplicate.values
test_y = test_X.is_duplicate.values

train_X = scaler.transform(np.nan_to_num(train_X[feature_col].as_matrix()))
test_X = scaler.transform(np.nan_to_num(test_X[feature_col].as_matrix()))



#train_y = train_X.is_duplicate.values
#train_X = train_X[feature_col] # remove weights for ensemble
#
#test_y = test_X.is_duplicate.values
#test_X = test_X[feature_col]
#
#
#
#train_X = coo_matrix(train_X)
#train_X = hstack([train_X])
#
#
#test_X = coo_matrix(test_X)
#test_X = hstack([test_X])





from sklearn.linear_model import SGDClassifier as SGD
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
#from sklearn.metrics import make_scorer

#f1_scorer = make_scorer(f1_score, pos_label="postive",average='binary')
#precision_scorer = make_scorer(precision_score, pos_label="postive")

sgd_params = {'alpha': [0.00006, 0.00007, 0.00008, 0.0001, 0.0005,0.0003]} # Regularization parameter
    
model_SGD = GridSearchCV(SGD(random_state = 2, 
                             shuffle = True, 
                             penalty='l2',
                             loss = 'log'), 
                             sgd_params, 
                             scoring='neg_log_loss',
                             #precision_scorer, 
                             cv = 10) # Find out which regularization parameter works the best. 
                            
model_SGD.fit(train_X, train_y) # Fit the model.
SGD_result = model_SGD.predict_proba(test_X)

log_loss(test_y,SGD_result)






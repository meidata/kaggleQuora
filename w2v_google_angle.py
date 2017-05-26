#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 21:40:26 2017

@author: meiyi
"""

import numpy as np
import pandas as pd
import platform


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



train_w2v_q1 = np.load(path_feature+'train_q1_w2v_google.pkl')

train_w2v_q2 = np.load(path_feature+'train_q2_w2v_google.pkl')

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



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


train_google_angle = []

for i in range(0,train_w2v_q1.shape[0]):
    train_google_angle.append(angle_between(train_w2v_q1[i],train_w2v_q2[i]))
    
np.save(path_feature+'train_google_angle.npy', train_google_angle,allow_pickle=True)
    
    
test_google_angle = []

for i in range(0,test_w2v_q1.shape[0]):
    test_google_angle.append(angle_between(test_w2v_q1[i],test_w2v_q2[i]))
    
    
np.save(path_feature+'test_google_angle.npy', test_google_angle,allow_pickle=True)


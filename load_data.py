"""
Functions to load the dataset.
"""

import numpy as np
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
           

def load():
    """Conveninence function to load all data as numpy arrays.
    """
    print("Loading data...")
    path_w2v,path_data,path_feature  = setPath()
    
    y_train = np.array(np.nan_to_num(np.load(path_feature +'train_is_duplicate.npy'))).astype(np.float32)
    X_train = np.array(np.load(path_feature +'train_feature_with_intersec_no_hash.npy')).astype(np.float32))
    X_test = np.array(np.nan_to_num(np.load(path_feature +'test_feature_with_intersec_no_hash.npy'))).astype(np.float32)
    return X_train, y_train, X_test

if __name__ == '__main__':np.array(np.nan_to_num(np.load(path_feature +'train_is_duplicate.npy'))).astype(np.float32)

    X_train, y_train, X_test = load()

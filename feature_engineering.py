"""
features generating

@author: Meiyi PAN
"""

import pickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
stop_words = stopwords.words('english')


path_w2v = '/Users/meiyi/Desktop/kaggle_quora/w2v_pretrained/'
model = gensim.models.KeyedVectors.load_word2vec_format(path_w2v+'GoogleNews-vectors-negative300.bin.gz', binary=True)
 
norm_model = gensim.models.KeyedVectors.load_word2vec_format(path_w2v+'GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)

path_feature = '/Users/meiyi/Desktop/kaggle_quora/features/'

def generateFS(data,file):
    def wmd(s1, s2):
        s1 = str(s1).lower().split()
        s2 = str(s2).lower().split()
        stop_words = stopwords.words('english')
        s1 = [w for w in s1 if w not in stop_words]
        s2 = [w for w in s2 if w not in stop_words]
        return model.wmdistance(s1, s2)
    
    
    def norm_wmd(s1, s2):
        s1 = str(s1).lower().split()
        s2 = str(s2).lower().split()
        stop_words = stopwords.words('english')
        s1 = [w for w in s1 if w not in stop_words]
        s2 = [w for w in s2 if w not in stop_words]
        return norm_model.wmdistance(s1, s2)
    
    
    def sent2vec(s):
        words = str(s).lower()
        words = word_tokenize(words)
        words = [w for w in words if not w in stop_words]
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(model[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())
    
    print('\n')
    

    
    '''
    Basic features:
    Length of question1
    Length of question2
    Difference in the two lengths
    Character length of question1 without spaces
    Character length of question2 without spaces
    Number of words in question1
    Number of words in question2
    Number of common words in question1 and question2
    '''
    print('..basic features..\n')
    data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
    data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
    data['diff_len'] = data.len_q1 - data.len_q2
    data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
    data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
    data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
    
    
    
    '''
    QRatio
    WRatio
    Partial ratio
    Partial token set ratio
    Partial token sort ratio
    Token set ratio
    Token sort ratio
    calculate a similarity score between two strings
    '''
    print('..fuzz features..\n')
    data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
       
#    data.to_pickle(path+'data.pkl')
#    data = pd.read_pickle(path+'data.pkl')
    

    print('..w2v features..\n')
    
    data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)    
    data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)
    
    question1_vectors = np.zeros((data.shape[0], 300))
    error_count = 0
    
    for i, q in tqdm(enumerate(data.question1.values)):
        question1_vectors[i, :] = sent2vec(q)
    
    question2_vectors  = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.question2.values)):
        question2_vectors[i, :] = sent2vec(q)
        
        
        
    print('..distances features..\n')
    data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]
    

    
    svd = TruncatedSVD(100)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    
    # Run SVD on the training data, then project the training data.
    w2v_q1_lsa = lsa.fit_transform(np.nan_to_num(question1_vectors))
    w2v_q2_lsa = lsa.fit_transform(np.nan_to_num(question2_vectors))
    
    
    np.save(file + '_w2v_q1_lsa.npy', w2v_q1_lsa,allow_pickle=True)
    np.save(file + '_w2v_q2_lsa.npy', w2v_q2_lsa,allow_pickle=True)



    
    print('..dumping features..\n')
    pickle.dump(question1_vectors, open(path_feature+file+'_q1_w2v_google.pkl', 'wb'), -1)
    pickle.dump(question2_vectors, open(path_feature+file+'_q2_w2v_google.pkl', 'wb'), -1)
    
#    data.to_csv(path+file+'_quora_features.csv', index=False)
    data.to_pickle(path_feature+file+'_quora_features.pkl')
    print('..done..')
    
    

    
    
path = '/Users/meiyi/Desktop/kaggle_quora/'
data = pd.read_csv(path+'test.csv')
data = data[[x for x in data.columns.values if 'id' not in x]]
data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9,data_10=np.array_split(data, 10)


for i in list(range(0,10)):
    print(i)
    temp = [data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9,data_10]
    generateFS(temp[i],'test_'+str(i))




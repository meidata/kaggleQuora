
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
pd.set_option('max_colwidth',-1)
seed = 1024
np.random.seed(seed)
path = '/Users/meiyi/Desktop/kaggle_quora/'

train = pd.read_csv(path+"train.csv")
test = pd.read_csv(path+"test.csv")


# In[22]:

'''
generate porter stemmatization
'''

def stem_str(x,stemmer=SnowballStemmer('english')):
    x = str(x).lower()
    x = text.re.sub("[^a-zA-Z0-9]"," ", x)
    x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
    x = " ".join(x.split())
    return x

snowball = SnowballStemmer('english')

print('Generate porter...\n')
train['question1_porter'] = train['question1'].apply(lambda x:stem_str(x))
test['question1_porter'] = test['question1'].apply(lambda x:stem_str(x))
train['question2_porter'] = train['question2'].apply(lambda x:stem_str(x))
test['question2_porter'] = test['question2'].apply(lambda x:stem_str(x))


pd.to_pickle(train[['question1_porter','question2_porter']],path+"lsa_xgb/train_porter.pkl")
pd.to_pickle(test[['question1_porter','question2_porter']],path+"lsa_xgb/test_porter.pkl")


# In[46]:

'''
generate tfidf vectors 
'''

from sklearn.feature_extraction.text import TfidfVectorizer

tf_vectorizer = TfidfVectorizer(min_df=3, 
                                max_features = None,
                                sublinear_tf=True,
                                use_idf=True,
                                analyzer = 'word',
                                ngram_range=(1,2))

def cleanText(x):
    x = str(x).lower()
    return x

# question_1 = train['question1'].apply(lambda x: cleanText(x)).tolist() + test['question1'].apply(lambda x: cleanText(x)).tolist() 
# question_2 = train['question2'].apply(lambda x: cleanText(x)).tolist() + test['question2'].apply(lambda x: cleanText(x)).tolist()

data_all = pd.concat([train,test])
questions = ['question1','question2']

corpus = []

for f in questions:
    data_all[f] = data_all[f].apply(lambda x: cleanText(x))
    corpus+=data_all[f].values.tolist()

tf_vectorizer.fit(corpus)

for f in questions:
    tfidfs = tf_vectorizer.transform(data_all[f].values.tolist())
    train_tfidf = tfidfs[:train.shape[0]]
    test_tfidf = tfidfs[train.shape[0]:]
    print('1..\n')
    pd.to_pickle(train_tfidf,path+'lsa_xgb/train_%s_tfidf.pkl'%f)
    print('2..\n')
    pd.to_pickle(test_tfidf,path+'lsa_xgb/test_%s_tfidf.pkl'%f)
    
corpus = []
questions_porter = ['question1_porter','question2_porter']

for f in questions_porter:
    data_all[f] = data_all[f].apply(lambda x: cleanText(x))
    corpus+=data_all[f].values.tolist()

tf_vectorizer.fit(corpus)

for f in questions_porter:
    tfidfs = tf_vectorizer.transform(data_all[f].values.tolist())
    train_tfidf = tfidfs[:train.shape[0]]
    test_tfidf = tfidfs[train.shape[0]:]
    print('1..\n')
    pd.to_pickle(train_tfidf,path+'lsa_xgb/train_%s_tfidf.pkl'%f)
    print('2..\n')
    pd.to_pickle(test_tfidf,path+'lsa_xgb/test_%s_tfidf.pkl'%f)
    


# In[4]:

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

train_question1_tfidf = pd.read_pickle('train_question1_tfidf.pkl')
train_question2_tfidf = pd.read_pickle('train_question2_tfidf.pkl')

train_question1_porter_tfidf = pd.read_pickle('train_question1_porter_tfidf.pkl')
train_question2_porter_tfidf = pd.read_pickle('train_question2_porter_tfidf.pkl')

test_question1_tfidf = pd.read_pickle('test_question1_tfidf.pkl')
test_question2_tfidf = pd.read_pickle('test_question2_tfidf.pkl')

test_question1_porter_tfidf = pd.read_pickle('test_question1_porter_tfidf.pkl')
test_question2_porter_tfidf = pd.read_pickle('test_question2_porter_tfidf.pkl')


svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))

# Run SVD on the training data, then project the training data.
X_train_question1_lsa = lsa.fit_transform(train_question1_tfidf)
X_train_question2_lsa = lsa.fit_transform(train_question2_tfidf)

# X_train_question1_porter_lsa = lsa.fit_transform(train_question1_porter_tfidf)
# X_train_question2_porter_lsa = lsa.fit_transform(train_question2_porter_tfidf)

X_test_question1_lsa = lsa.fit_transform(test_question1_tfidf)
X_test_question2_lsa = lsa.fit_transform(test_question2_tfidf)

# X_test_question1_porter_lsa = lsa.fit_transform(test_question1_porter_tfidf)
# X_test_question2_porter_lsa = lsa.fit_transform(test_question2_porter_tfidf)

np.save('X_train_question1_lsa.npy', X_train_question1_lsa,allow_pickle=True)
np.save('X_train_question2_lsa.npy', X_train_question2_lsa,allow_pickle=True)
np.save('X_test_question1_lsa.npy', X_test_question1_lsa,allow_pickle=True)
np.save('X_test_question2_lsa.npy', X_test_question2_lsa,allow_pickle=True)


# In[95]:

import numpy as np
import scipy.spatial.distance
from sklearn.metrics.pairwise import cosine_similarity

def sentenceCosinesimlarity(q1,q2):
    similarity = []
    for i in range(0,len(q1)):
        if np.isnan(q1[i][0]) or np.isnan(q2[i][0]):
            similarity.append(-1)
        else:
            temp = cosine_similarity(q1[i].reshape(1, -1),q2[i].reshape(1, -1))
            similarity.append(temp[0][0])   
    df_similarity = pd.DataFrame(similarity,columns=['cosineSimilarity'])
    return df_similarity


pd.to_pickle(sentenceCosinesimlarity(X_train_question1_lsa,X_train_question2_lsa),
             path+'lsa_xgb/train_cosine_similarity.pkl')

pd.to_pickle(sentenceCosinesimlarity(X_train_question1_porter_lsa,X_train_question2_porter_lsa),
             path+'lsa_xgb/train_porter_cosine_similarity.pkl')

pd.to_pickle(sentenceCosinesimlarity(X_test_question1_lsa,X_test_question2_lsa),
             path+'lsa_xgb/test_cosine_similarity.pkl')

pd.to_pickle(sentenceCosinesimlarity(X_test_question1_porter_lsa,X_test_question2_porter_lsa),
             path+'lsa_xgb/test_porter_cosine_similarity.pkl')


# In[12]:

import pickle
train_cosine_similarity = pd.read_pickle('train_cosine_similarity.pkl')
train_porter_cosine_similarity = pd.read_pickle('train_porter_cosine_similarity.pkl')

test_cosine_similarity = pd.read_pickle('test_cosine_similarity.pkl')
test_porter_cosine_similarity = pd.read_pickle('test_porter_cosine_similarity.pkl')


'''
read and concat
'''
with open(path+'lsa_xgb/train_porter.pkl', 'rb') as handle:
    train_porter = pd.DataFrame(pickle.load(handle),columns=['question1_porter','question2_porter']) 
    
with open(path+'lsa_xgb/test_porter.pkl', 'rb') as handle:
    test_porter = pd.DataFrame(pickle.load(handle),columns=['question1_porter','question2_porter']) 
    
train = pd.concat([train,train_porter],axis = 1)
test= pd.concat([test,test_porter],axis = 1)



# In[112]:

import numpy as np

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


def sentenceAngle(q1,q2):
    angle = []
    for i in range(0,len(q1)):
        if np.isnan(q1[i][0]) or np.isnan(q2[i][0]):
            angle.append(360)
        else:
            temp = angle_between(q1[i],q2[i])
            angle.append(temp)   
            
    df_angle = pd.DataFrame(angle,columns=['cosAngle'])
    return df_angle

pd.to_pickle(sentenceAngle(X_train_question1_lsa,X_train_question2_lsa),
             path+'lsa_xgb/train_cosine_angle.pkl')

pd.to_pickle(sentenceAngle(X_train_question1_porter_lsa,X_train_question2_porter_lsa),
             path+'lsa_xgb/train_porter_cosine_angle.pkl')

pd.to_pickle(sentenceAngle(X_test_question1_lsa,X_test_question2_lsa),
             path+'lsa_xgb/test_cosine_angle.pkl')

pd.to_pickle(sentenceAngle(X_test_question1_porter_lsa,X_test_question2_porter_lsa),
             path+'lsa_xgb/test_porter_cosine_angle.pkl')


# In[13]:

import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy import sparse as ssp
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import distance
stop_words = stopwords.words('english')
    
#stops = set(stopwords.words("english"))
stops = set(["http","www","img","border","home","body","a","about","above","after","again","against","all","am","an",
"and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't",
"cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from",
"further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers",
"herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
"itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought",
"our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such",
"than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're",
"they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were",
"weren't","what","what's","when","when's""where","where's","which","while","who","who's","whom","why","why's","with","won't","would",
"wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves" ])
porter = PorterStemmer()
snowball = SnowballStemmer('english')

weights={}

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=5000.0, min_count=2.0):
    if count < min_count:
        return 0.0
    else:
        return 1.0 / (count + eps)


def word_shares(row,wei,stop):
    q1 = set(str(row['question1']).lower().split())
    q1words = q1.difference(stop)
    if len(q1words) == 0:
        return '0:0:0:0:0'

    q2 = set(str(row['question2']).lower().split())
    q2words = q2.difference(stop)
    if len(q2words) == 0:
        return '0:0:0:0:0'

    q1stops = q1.intersection(stop)
    q2stops = q2.intersection(stop)

    shared_words = q1words.intersection(q2words)
    #print(len(shared_words))
    shared_weights = [wei.get(w, 0) for w in shared_words]
    total_weights = [wei.get(w, 0) for w in q1words] + [wei.get(w, 0) for w in q2words]

    R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
    R2 = float(len(shared_words)) / (float(len(q1words)) + float(len(q2words))) #count share
    R31 = float(len(q1stops)) / float(len(q1words)) #stops in q1
    R32 = float(len(q2stops)) / float(len(q2words)) #stops in q2
    return '{}:{}:{}:{}:{}'.format(R1, R2, float(len(shared_words)), R31, R32)

def stem_str(x,stemmer=SnowballStemmer('english')):
        x = text.re.sub("[^a-zA-Z0-9]"," ", x)
        x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
        x = " ".join(x.split())
        return x
    
def calc_set_intersection(text_a, text_b):
    try:
        a = set(text_a.split())
        b = set(text_b.split())
        return len(a.intersection(b)) *1.0 / len(a)
    except ZeroDivisionError:
        print("divide by zero\n")

def str_abs_diff_len(str1, str2):
    return abs(len(str1)-len(str2))

def str_len(str1):
    return len(str(str1))

def char_len(str1):
    str1_list = set(str(str1).replace(' ',''))
    return len(str1_list)

def word_len(str1):
    str1_list = str1.split(' ')
    return len(str1_list)

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stop_words:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stop_words:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))*1.0/(len(q1words) + len(q2words))
    return R

def str_jaccard(str1, str2):
    str1_list = str1.split(" ")
    str2_list = str2.split(" ")
    res = distance.jaccard(str1_list, str2_list)
    return res

# shortest alignment
def str_levenshtein_1(str1, str2):
    #str1_list = str1.split(' ')
    #str2_list = str2.split(' ')
    res = distance.nlevenshtein(str1, str2,method=1)
    return res

# longest alignment
def str_levenshtein_2(str1, str2):

    #str1_list = str1.split(' ')
    #str2_list = str2.split(' ')
    res = distance.nlevenshtein(str1, str2,method=2)
    return res

def str_sorensen(str1, str2):

    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    res = distance.sorensen(str1_list, str2_list)
    return res


test['is_duplicated']=[-1]*test.shape[0]

print('Generate intersection')
train_interaction = train.astype(str).apply(lambda x:calc_set_intersection(x['question1'],x['question2']),axis=1)
test_interaction = test.astype(str).apply(lambda x:calc_set_intersection(x['question1'],x['question2']),axis=1)
pd.to_pickle(train_interaction,path+"train_interaction.pkl")
pd.to_pickle(test_interaction,path+"test_interaction.pkl")

print('Generate porter intersection')
train_porter_interaction = train.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)
test_porter_interaction = test.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)

pd.to_pickle(train_porter_interaction,path+"train_porter_interaction.pkl")
pd.to_pickle(test_porter_interaction,path+"test_porter_interaction.pkl")  
    

        
##################### generate_len.py #########################



print('Generate len')
feats = []

train['abs_diff_len'] = train.astype(str).apply(lambda x:str_abs_diff_len(x['question1'],x['question2']),axis=1)
test['abs_diff_len']= test.astype(str).apply(lambda x:str_abs_diff_len(x['question1'],x['question2']),axis=1)
feats.append('abs_diff_len')

train['R']=train.apply(word_match_share, axis=1, raw=True)
test['R']=test.apply(word_match_share, axis=1, raw=True)
feats.append('R')

train['common_words'] = train.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
test['common_words'] = test.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
feats.append('common_words')

for c in ['question1','question2']:
    train['%s_char_len'%c] = train[c].astype(str).apply(lambda x:char_len(x))
    test['%s_char_len'%c] = test[c].astype(str).apply(lambda x:char_len(x))
    feats.append('%s_char_len'%c)

    train['%s_str_len'%c] = train[c].astype(str).apply(lambda x:str_len(x))
    test['%s_str_len'%c] = test[c].astype(str).apply(lambda x:str_len(x))
    feats.append('%s_str_len'%c)

    train['%s_word_len'%c] = train[c].astype(str).apply(lambda x:word_len(x))
    test['%s_word_len'%c] = test[c].astype(str).apply(lambda x:word_len(x))
    feats.append('%s_word_len'%c)


pd.to_pickle(train[feats].values,path+"train_len.pkl")
pd.to_pickle(test[feats].values,path+"test_len.pkl")       

#########################generate_distance.py #################

test['is_duplicated']=[-1]*test.shape[0]

data_all = pd.concat([train,test])    

print('Generate jaccard')
train_jaccard = train.astype(str).apply(lambda x:str_jaccard(x['question1'],x['question2']),axis=1)
test_jaccard = test.astype(str).apply(lambda x:str_jaccard(x['question1'],x['question2']),axis=1)
pd.to_pickle(train_jaccard,path+"train_jaccard.pkl")
pd.to_pickle(test_jaccard,path+"test_jaccard.pkl")

print('Generate porter jaccard')
train_porter_jaccard = train.astype(str).apply(lambda x:str_jaccard(x['question1_porter'],x['question2_porter']),axis=1)
test_porter_jaccard = test.astype(str).apply(lambda x:str_jaccard(x['question1_porter'],x['question2_porter']),axis=1)

pd.to_pickle(train_porter_jaccard,path+"train_porter_jaccard.pkl")
pd.to_pickle(test_porter_jaccard,path+"test_porter_jaccard.pkl")  



# In[2]:

X_train_question1_lsa = np.load('X_train_question1_lsa.npy')
X_train_question2_lsa = np.load('X_train_question2_lsa.npy')
X_test_question1_lsa = np.load('X_test_question1_lsa.npy')
X_test_question2_lsa = np.load('X_test_question2_lsa.npy')

X_train_question1_lsa = pd.DataFrame(X_train_question1_lsa, columns=['q1_' + i for i in list(map(str,range(0,X_train_question1_lsa.shape[1])))])
X_train_question2_lsa = pd.DataFrame(X_train_question2_lsa, columns=['q2_' + i for i in list(map(str,range(0,X_train_question2_lsa.shape[1])))])


X_test_question1_lsa = pd.DataFrame(X_test_question1_lsa, columns=['q1_' + i for i in list(map(str,range(0,X_test_question1_lsa.shape[1])))])
X_test_question2_lsa = pd.DataFrame(X_test_question2_lsa, columns=['q2_' + i for i in list(map(str,range(0,X_test_question2_lsa.shape[1])))])



train_cosine_similarity = pd.DataFrame(pd.read_pickle('train_cosine_similarity.pkl'),columns=['cosSim'])
train_porter_cosine_similarity = pd.DataFrame(pd.read_pickle('train_porter_cosine_similarity.pkl'),columns=['cosSimPorter'])

test_cosine_similarity = pd.DataFrame(pd.read_pickle('test_cosine_similarity.pkl'),columns=['cosSim'])
test_porter_cosine_similarity = pd.DataFrame(pd.read_pickle('test_porter_cosine_similarity.pkl'),columns=['cosSimPorter'])


train_interaction = pd.read_pickle(path+'train_interaction.pkl')
train_interaction = np.nan_to_num(train_interaction)
train_interaction = pd.DataFrame(train_interaction,columns=['interaction'])

test_interaction = pd.read_pickle(path+'test_interaction.pkl')
test_interaction = np.nan_to_num(test_interaction)
test_interaction = pd.DataFrame(test_interaction,columns=['interaction'])


train_porter_interaction = pd.read_pickle(path+'train_porter_interaction.pkl')
train_porter_interaction = np.nan_to_num(train_porter_interaction)
train_porter_interaction = pd.DataFrame(train_porter_interaction,columns=['interactionPorter'])

test_porter_interaction = pd.read_pickle(path+'test_porter_interaction.pkl')
test_porter_interaction = np.nan_to_num(test_porter_interaction)
test_porter_interaction = pd.DataFrame(test_porter_interaction,columns=['interactionPorter'])


train_jaccard = pd.read_pickle(path+'train_jaccard.pkl')
train_jaccard = np.nan_to_num(train_jaccard)
train_jaccard = pd.DataFrame(train_jaccard,columns=['jaccard'])

test_jaccard = pd.read_pickle(path+'test_jaccard.pkl')
test_jaccard = np.nan_to_num(test_jaccard)
test_jaccard = pd.DataFrame(test_jaccard,columns=['jaccard'])


train_porter_jaccard = pd.read_pickle(path+'train_porter_jaccard.pkl')
train_porter_jaccard = np.nan_to_num(train_porter_jaccard)
train_porter_jaccard = pd.DataFrame(train_porter_jaccard,columns=['porterJaccard'])

test_porter_jaccard = pd.read_pickle(path+'test_porter_jaccard.pkl')
test_porter_jaccard = np.nan_to_num(test_porter_jaccard)
test_porter_jaccard = pd.DataFrame(test_porter_jaccard,columns=['porterJaccard'])


train_cosine_angle = pd.DataFrame(pd.read_pickle('train_cosine_angle.pkl'),columns=['cosAngle'])
train_porter_cosine_angle = pd.DataFrame(pd.read_pickle('train_porter_cosine_angle.pkl'),columns=['cosAnglePorter'])

test_cosine_angle = pd.DataFrame(pd.read_pickle('test_cosine_angle.pkl'),columns=['cosAngle'])

test_porter_cosine_angle = pd.DataFrame(pd.read_pickle('test_porter_cosine_angle.pkl'),columns=['cosAnglePorter'])


train_len = pd.read_pickle(path+"train_len.pkl")
test_len = pd.read_pickle(path+"test_len.pkl")

train_len=np.nan_to_num(train_len)
test_len=np.nan_to_num(test_len) 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(np.vstack([train_len,test_len]))
train_len = scaler.transform(train_len)
test_len =scaler.transform(test_len)

train_len = pd.DataFrame(train_len, columns=['len_' + i for i in list(map(str,range(0,train_len.shape[1])))])
test_len = pd.DataFrame(test_len, columns=['len_' + i for i in list(map(str,range(0,test_len.shape[1])))])



# In[ ]:


train_1 = pd.concat([train,X_train_question1_lsa,
                   X_train_question2_lsa,
                   train_cosine_similarity,
                   train_interaction,
                   train_porter_interaction,
                   train_cosine_angle,
                   train_porter_cosine_angle,
                   train_len,
                   train_porter_jaccard,
                   train_jaccard],axis = 1)


test_1 = pd.concat([test,X_test_question1_lsa,
                   X_test_question2_lsa,
                   test_cosine_similarity,
                   test_interaction,
                   test_porter_interaction,
                   test_cosine_angle,
                   test_porter_cosine_angle,
                   test_len,
                   test_porter_jaccard,
                   test_jaccard],axis = 1)


# In[11]:

from sklearn.cross_validation import train_test_split

feature_col = train_1.columns.difference(['id','qid1','qid2','question1','question2','question1_porter','question2_porter']).values.tolist()

pos_train, pos_test = train_test_split(train_1.ix[train_1['is_duplicate']==1,feature_col], test_size = 0.3)
neg_train, neg_test = train_test_split(train_1.ix[train_1['is_duplicate']==0,feature_col], test_size = 0.3)

train_X = pos_train.append(neg_train)
test_X = pos_test.append(neg_test)

train_y = train_X.is_duplicate.values
train_X = train_X[train_X.columns.difference(['is_duplicate'])]

test_y = test_X.is_duplicate.values
test_X = test_X[test_X.columns.difference(['is_duplicate'])]


# In[215]:

import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(train_X, label=train_y)
d_valid = xgb.DMatrix(test_X, label=test_y)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

a = 0.165 / 0.37

b = (1 - 0.165) / (1 - 0.37)

def kappa(preds, y):
    score = []
    for pp,yy in zip(preds, y.get_label()):
        score.append(a * yy * np.log (pp) + b * (1 - yy) * np.log(1-pp))
    score = -np.sum(score) / len(score)

    return 'kappa', float(score)

bst = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=5, verbose_eval=10, feval = kappa)


# In[12]:

bst.save_model('1000_034.model')


# In[13]:

feature_col_test = train_X.columns.values.tolist()

d_test = xgb.DMatrix(test_1.ix[:,feature_col_test])
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb_2504.csv', index=False)


# In[ ]:





# coding: utf-8

# In[ ]:

'''
get features created in the w2v model
'''

# In[11]:

from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context_bigram")



# In[12]:

def makeFeatureVec(question, model, num_features):
    
    featureVec = np.zeros((num_features),dtype="float32")
    #
    nwords = 0.
    # 
    index2word_set = set(model.wv.index2word)
    #
    for word in question:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
            
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(questions, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    questionFeatureVecs = np.zeros((len(questions),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for question in questions:
       
       # Print a status message every 1000th review
        if counter%1000. == 0.:
            print("Question %d of %d" % (counter, len(questions)))
       
       # Call the function (defined above) that makes average feature vectors
        questionFeatureVecs[counter] = makeFeatureVec(question, model, num_features)
       
       # Increment the counter
        counter = counter + 1
    return questionFeatureVecs


# In[141]:

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.
np.seterr(divide='ignore')
def sentencesToWordsRmStops(questions,remove_stopwords = True):
    tbw = TreebankWordTokenizer()
    question = str(questions)
    q_1 = question.lower()
    q_1 = re.sub('\?','',q_1)
#     question = re.sub("[^a-zA-Z0-9]",' ',question) 
    q_2 = tbw.tokenize(q_1)
    wnl = WordNetLemmatizer()
    q_3 = [wnl.lemmatize(i,pos='v') for i in q_2 if len(i) > 1]
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        question = [w for w in q_3 if not w in stops]
    return q_3


num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3  

tic()
rmStopwords_clean_train = list(map(sentencesToWordsRmStops,train_question))
print('done..cleaning...texts..\n')
trainDataVecs = getAvgFeatureVecs(rmStopwords_clean_train, model, num_features)
toc()

print("Creating average feature vecs for test reviews")
tic()
rmStopwords_clean_test = list(map(sentencesToWordsRmStops,test_question))
print('done..cleaning...texts..\n')
testDataVecs = getAvgFeatureVecs(rmStopwords_clean_test, model, num_features)
toc()



# In[ ]:

'''
Train word2vec model
'''


# In[123]:

import pandas as pd
import numpy as np
pd.set_option('max_colwidth',-1)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer


# In[2]:

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[ ]:

'''
calculate cosine similarity of two sentences by the features vectors of w2v
'''


# In[34]:

import numpy as np
import scipy.spatial.distance
from sklearn.metrics.pairwise import cosine_similarity


# In[126]:

train_q1_q2_cosine = []
count = 0
for i in range(0,int(len(trainDataVecs)/2)):
    if np.isnan(trainDataVecs[i][0]) or np.isnan(trainDataVecs[404290+i][0]):
        train_q1_q2_cosine.append(0.00)
        count += 1 
    else:
        print(i)
        temp = cosine_similarity(trainDataVecs[i].reshape(1, -1),trainDataVecs[404290+i].reshape(1, -1), dense_output=True)
        train_q1_q2_cosine.append(temp[0][0])
        
with open('train_q1_q2_cosine.pickle', 'wb') as handle:
    pickle.dump(train_q1_q2_cosine, handle, protocol=pickle.HIGHEST_PROTOCOL)      


# In[132]:

test_q1_q2_cosine = []
count = 0
for i in range(0,int(len(testDataVecs)/2)):
    if np.isnan(testDataVecs[i][0]) or np.isnan(testDataVecs[2345796+i][0]):
        test_q1_q2_cosine.append(0.00)
        count += 1 
    else:
        print(i)
        temp = cosine_similarity(testDataVecs[i].reshape(1, -1),testDataVecs[2345796+i].reshape(1, -1), dense_output=True)
        test_q1_q2_cosine.append(temp[0][0])
        
with open('test_q1_q2_cosine.pickle', 'wb') as handle:
    pickle.dump(test_q1_q2_cosine, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        



# In[ ]:

'''
get the angle bwt two vectors 
'''

# get the angle bwt two question
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

train_q1_q2_angle = []
count = 0
for i in range(0,int(len(trainDataVecs)/2)):
    if np.isnan(trainDataVecs[i][0]) or np.isnan(trainDataVecs[404290+i][0]):
        train_q1_q2_angle.append(360)
        count += 1 
    else:
        print(i)
        train_q1_q2_angle.append(angle_between(trainDataVecs[i],trainDataVecs[404290+i]))

        
with open('train_q1_q2_angle.pickle', 'wb') as handle:
    pickle.dump(train_q1_q2_angle, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    

test_q1_q2_angle = []
count = 0
for i in range(0,int(len(testDataVecs)/2)):
    if np.isnan(testDataVecs[i][0]) or np.isnan(testDataVecs[2345796+i][0]):
        test_q1_q2_angle.append(360)
        count += 1 
    else:
        print(i)
        test_q1_q2_angle.append(angle_between(testDataVecs[i],testDataVecs[2345796+i]))


with open('test_q1_q2_angle.pickle', 'wb') as handle:
    pickle.dump(test_q1_q2_angle, handle, protocol=pickle.HIGHEST_PROTOCOL)  
                               


# In[123]:

'''
calculate the cosine distance

'''
from scipy.spatial.distance import cosine


train_q1_q2_new_cosine = []
count = 0
for i in range(0,int(len(trainDataVecs)/2)):
    if np.isnan(trainDataVecs[i][0]) or np.isnan(trainDataVecs[404290+i][0]):
        train_q1_q2_new_cosine.append(-1.)
        count += 1 
    else:
        print(i)
        train_q1_q2_new_cosine.append(cosine(trainDataVecs[i].astype(np.float16),trainDataVecs[404290+i].astype(np.float16)))

        
with open('train_q1_q2_new_cosine.pickle', 'wb') as handle:
    pickle.dump(train_q1_q2_new_cosine, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    
test_q1_q2_new_cosine = []
count = 0
for i in range(0,int(len(testDataVecs)/2)):
    if np.isnan(testDataVecs[i][0]) or np.isnan(testDataVecs[2345796+i][0]):
        test_q1_q2_new_cosine.append(360)
        count += 1 
    else:
        print(i)
        test_q1_q2_new_cosine.append(angle_between(testDataVecs[i],testDataVecs[2345796+i]))

                
with open('test_q1_q2_new_cosine.pickle', 'wb') as handle:
    pickle.dump(test_q1_q2_new_cosine, handle, protocol=pickle.HIGHEST_PROTOCOL)  


# In[246]:

train_w2vectors = []
count = 0
for i in range(0,int(len(trainDataVecs)/2)):
    if np.isnan(trainDataVecs[i][0]) or np.isnan(trainDataVecs[404290+i][0]):
        train_q1_q2_cosine.append(0.00)
        count += 1 
    else:

        temp = abs(trainDataVecs[i] - trainDataVecs[404290+i])
        train_w2vectors.append(temp)
        
test_w2vectors = []
count = 0
for i in range(0,int(len(testDataVecs)/2)):
    if np.isnan(testDataVecs[i][0]) or np.isnan(testDataVecs[2345796+i][0]):
        test_q1_q2_cosine.append(0.00)
        count += 1 
    else:
        print(i)
        temp = abs(testDataVecs[i]- testDataVecs[2345796+i])
        test_w2vectors.append(temp)



# In[253]:

train_w2v = pd.DataFrame(train_w2vectors)


# In[255]:

train_w2v.head(4)


# In[ ]:

'''
merge the feautres into the train and test data frame

'''

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sampleSub = pd.read_csv('sample_submission.csv')

import pickle
with open('test_q1_q2_cosine.pickle', 'rb') as handle:
    test_q1_q2_cosine = pickle.load(handle) 
    
with open('train_q1_q2_cosine.pickle', 'rb') as handle:
    train_q1_q2_cosine = pickle.load(handle) 
    
with open('train_q1_q2_new_cosine.pickle', 'rb') as handle:
    train_new_cosine = pickle.load(handle) 

with open('test_q1_q2_new_cosine.pickle', 'rb') as handle:
    test_new_cosine = pickle.load(handle) 
    
    
train_Vec = pd.DataFrame(train_q1_q2_cosine, columns=['cosSimilarity'])
train_w2v = pd.DataFrame(train_q1_q2_cosine, columns=['cosSimilarity'])
train_new_cosine = pd.DataFrame(train_q1_q2_new_cosine, columns=['cosDistance'])
train_angle = pd.DataFrame(train_q1_q2_angle, columns=['vecAngle'])

test_Vec = pd.DataFrame(test_q1_q2_cosine, columns=['cosSimilarity'])
test_new_cosine = pd.DataFrame(test_q1_q2_new_cosine, columns=['cosDistance'])
test_angle = pd.DataFrame(test_q1_q2_angle, columns=['vecAngle'])


train = pd.concat([train,train_Vec,train_angle,train_new_cosine ], axis=1)
test = pd.concat([test,test_Vec,test_angle,test_new_cosine ], axis=1)

train['dotVec'] = (train['cosSimilarity']) * (1/(train['vecAngle']+0.00000001))
test['dotVec'] = (test['cosSimilarity']) * (1/(test['vecAngle']+0.00000001))


# In[170]:

def preProcessQuestion(question):
    q_1 = str(question).lower()
    q_1 = re.sub('\?','',q_1)
    wnl = WordNetLemmatizer()
    row_2 = [wnl.lemmatize(i,pos='v') for i in q_1.split() ]
    return ' '.join(row_2)

train['question1'] =  train['question1'].apply(lambda x : preProcessQuestion(x))
train['question2'] =  train['question2'].apply(lambda x : preProcessQuestion(x))

test['question1'] =  test['question1'].apply(lambda x : preProcessQuestion(x))
test['question2'] =  test['question2'].apply(lambda x : preProcessQuestion(x))


# In[178]:

train['q1len'] = train['question1'].str.len()
train['q2len'] = train['question2'].str.len()

def countNumber(row):
    row = str(row)
    return len(row.split(" "))

train['q1_n_words'] = train['question1'].apply(lambda row: len(row.split(" ")))
train['q2_n_words'] = train['question2'].apply(countNumber)


train['question2'] = train['question2'].apply(str)

def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))


train['word_share'] = train.apply(normalized_word_share, axis=1)



# In[222]:

test['q1len'] = test['question1'].str.len()
test['q2len'] = test['question2'].str.len()

def countNumber(row):
    row = str(row)
    return len(row.split(" "))

test['q1_n_words'] = test['question1'].apply(lambda row: len(row.split(" ")))
test['q2_n_words'] = test['question2'].apply(countNumber)


test['question2'] = test['question2'].apply(str)

def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))


test['word_share'] = test.apply(normalized_word_share, axis=1)



# In[204]:

from sklearn.model_selection import train_test_split

feature_col = ['is_duplicate','cosSimilarity', 'vecAngle', 'cosDistance', 'dotVec', 'q1len', 'q2len','q1_n_words', 'q2_n_words', 'word_share']
pos_train, pos_test = train_test_split(train.ix[train['is_duplicate']==1,feature_col], test_size = 0.3)
neg_train, neg_test = train_test_split(train.ix[train['is_duplicate']==0,feature_col], test_size = 0.3)

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

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)


# In[237]:

feature_col_test = ['cosDistance','cosSimilarity','dotVec', 'q1_n_words','q1len',  'q2_n_words','q2len', 'vecAngle',  'word_share']

d_test = xgb.DMatrix(test.ix[:,feature_col_test])
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb.csv', index=False)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




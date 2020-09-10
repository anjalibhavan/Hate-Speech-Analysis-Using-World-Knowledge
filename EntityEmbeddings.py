import tqdm
import numpy as np
import _pickle as cPickle
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Loading model
f = open('./pickles/dmmodel.pkl','rb')
print('loading model ...')
model = cPickle.load(f)

# Loading data
f = open('./pickles/dande_train.pkl','rb')
entities_tr = cPickle.load(f)
f = open('./pickles/dande_test.pkl','rb')
entities_tst = cPickle.load(f)

embeddings_tr = []
embeddings_tst = []

# Generating entity embeddings
for ent in tqdm.tqdm(entities_tr):
    temp = []
    for term in ent:
        temp.append(model.infer_vector([term]))
    embeddings_tr.append(temp)  

for ent in tqdm.tqdm(entities_tst):
    temp = []
    for term in ent:
        temp.append(model.infer_vector([term]))
    embeddings_tst.append(temp)  

features_tr = []
features_tst = []

# Stacking embeddings to generate trainable feature sets
for tweet_embeddings in embeddings_tr:
    temp = []
    for v in tweet_embeddings:
        temp.append(v.reshape((-1,1)))
    if len(tweet_embeddings) is 0:
        zero_vector = np.zeros((1,200))
        temp.append(zero_vector)
    averaged_vector = np.mean(temp,axis = 0)
    features_tr.append(averaged_vector)

for tweet_embeddings in embeddings_tst:
    temp = []
    for v in tweet_embeddings:
        temp.append(v.reshape((-1,1)))
    if len(tweet_embeddings) is 0:
        zero_vector = np.zeros((1,200))
        temp.append(zero_vector)
    averaged_vector = np.mean(temp,axis = 0)
    features_tst.append(averaged_vector)

features_tr = np.concatenate(features_tr, axis = 0)    
features_tst = np.concatenate(features_tst, axis = 0)

# Saving files
with open('./pickles/wiki_train.pkl','wb') as f:
    cPickle.dump(features_tr, f)

with open('./pickles/wiki_test.pkl','wb') as f:
    cPickle.dump(features_tst, f)

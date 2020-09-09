from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import torch
import numpy as np
import tqdm
import multiprocessing
import _pickle as cPickle

f = open('./pickles/dmmodel.pkl','rb')
print('loading model ...')
model = cPickle.load(f)

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

features = []

for i, embeds in enumerate(embeddings):
    temp = []
    for arr in embeds:
        torcharr = torch.from_numpy(arr)
        torcharr = torcharr.unsqueeze(0)
        torcharr = torcharr.numpy()
        temp.append(torcharr)
    if(len(embeds)==0):
        z = np.zeros((1,200))
        temp.append(z)
    res = np.mean(temp,axis = 0)
    features.append(res)

features = np.concatenate(features, axis = 0)
with open('./pickles/caa_embed.pkl','wb') as f:
    cPickle.dump(features,f)


from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
import torch
import numpy as np
import wikipedia
from wikipedia2vec import Wikipedia2Vec
import tqdm
from gensim.test.utils import get_tmpfile
from datetime import datetime
import multiprocessing
import _pickle as cPickle

f = open('./pickles/dmmodel.pkl','rb')
print('loading model ...')
model = cPickle.load(f)

f = open('./pickles/caa_dande.pkl','rb')
print('loading extracted entities ...')
entities = cPickle.load(f)

embeddings = []
for ent in tqdm.tqdm(entities):
    temp = []
    for term in ent:
        temp.append(model.infer_vector([term]))
    embeddings.append(temp)  

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

print(type(features[0]))
features = np.concatenate(features, axis = 0)
print(features.shape)
with open('./pickles/caa_embed.pkl','wb') as f:
    cPickle.dump(features,f)


from dandelion import default_config
from DataReader import DataReader
from TweetNormalizer import normalizeTweet
import torch
import tqdm
import _pickle as cPickle
from pprint import pprint
from dandelion import DataTXT

default_config['token'] = '48ef0a0b8f49467ea88bdfc69c9968ec'
datatxt = DataTXT()
'''
dr_tst = DataReader('./Data/testset-levela.tsv','A')
tst_data,tst_label = dr_tst.get_test_data()

assert len(tst_data) == len(tst_label)

data = tst_data[:]
labels = tst_label[:]

'''
dr = DataReader('./filtered.csv','A')
data,labels = dr.get_labelled_data()
data = data[:]

entities = []
for line in tqdm.tqdm(data):
    temp = []
    for annotation in datatxt.nex(normalizeTweet(line),lang='en').annotations:
        temp.append(annotation.title)
    entities.append(temp)
    
with open('./pickles/caa_dande.pkl','wb') as f:
    cPickle.dump(entities,f)

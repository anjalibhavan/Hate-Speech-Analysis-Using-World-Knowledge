import tqdm
import _pickle as cPickle
from DataReader import DataReader
from dandelion import DataTXT
from dandelion import default_config
from TweetNormalizer import normalizeTweet

# Initializing Dandelion API (can be obtained from https://dandelion.eu/)
default_config['token'] = 'INSERT TOKEN'
datatxt = DataTXT()

# Loading data
dr_tr = DataReader('./Data/olid-training-v1.tsv','A')
data_tr, labels_tr = dr_tr.get_labelled_data()
dr_tst = DataReader('./Data/testset-levela.tsv','A')
data_tst,label_tst = dr_tst.get_test_data()

data_tr = data_tr[:]
data_tst = data_tst[:]

entities_tr = []
entities_tst = []

# Entity extraction using dandelion
for line in tqdm.tqdm(data_tr):
    temp = []
    for annotation in datatxt.nex(normalizeTweet(line),lang='en').annotations:
        temp.append(annotation.title)
    entities_tr.append(temp)

for line in tqdm.tqdm(data_tst):
    temp = []
    for annotation in datatxt.nex(normalizeTweet(line),lang='en').annotations:
        temp.append(annotation.title)
    entities_tst.append(temp)

# Saving to files    
with open('./pickles/dande_train.pkl','wb') as f:
    cPickle.dump(entities_tr,f)
with open('./pickles/dande_test.pkl','wb') as f:
    cPickle.dump(entities_tst,f)

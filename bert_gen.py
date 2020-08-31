import torch
import argparse
from transformers import RobertaConfig
from transformers import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from DataReader import DataReader
from Preprocessor import Preprocessor
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
from TweetNormalizer import normalizeTweet

config = RobertaConfig.from_pretrained(
    "./BERTweet_base_transformers/config.json"
)
BERTweet = RobertaModel.from_pretrained(
    "./BERTweet_base_transformers/model.bin",
    config=config
)

parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="./BERTweet_base_transformers/bpe.codes",
    required=False,
    type=str,  
    help='path to fastBPE BPE'
)
args = parser.parse_args()
bpe = fastBPE(args)
 
vocab = Dictionary()
vocab.add_from_file("./BERTweet_base_transformers/dict.txt")

# Train set generation 

dr = DataReader('./filtered.csv','A')
data,labels = dr.get_labelled_data()
#data,labels = dr.shuffle(data,labels,'random')

data = data[:]
labels = labels[:]
print(len(data))
print(len(labels))

lines = [normalizeTweet(line) for line in data]
embeddings = []
print('We are here')
for line in tqdm(lines,'Generating train embeddings'):
    subword = '<s> ' + bpe.encode(line) + ' </s>'
    input_id = vocab.encode_line(subword, append_eos=False, add_if_not_exist=False).long().tolist()
    all_input_ids = torch.tensor([input_id], dtype=torch.long) 
    
    with torch.no_grad():  
        features = BERTweet(all_input_ids)  

    embeddings.append(torch.mean(features[0][0], dim=0))

stacked_tensor = torch.stack(embeddings)

X_train = torch.FloatTensor(stacked_tensor)
print(X_train.shape)
print(len(labels))

#y_train = torch.FloatTensor(labels)

with open('pickles/caa_mean.pkl','wb') as f:
    cPickle.dump((X_train, labels),f)

# Test set generation    
'''
dr_tst = DataReader('./Data/testset-levela.tsv','A')
tst_data,tst_label = dr_tst.get_test_data()

assert len(tst_data) == len(tst_label)

data = tst_data[:]
labels = tst_label[:]

lines = [normalizeTweet(line) for line in data]
embeddings = []

for line in tqdm(lines,'Generating test embeddings'):
    subword = '<s> ' + bpe.encode(line) + ' </s>'
    input_id = vocab.encode_line(subword, append_eos=False, add_if_not_exist=False).long().tolist()
    all_input_ids = torch.tensor([input_id], dtype=torch.long) 
    
    with torch.no_grad():  
        features = BERTweet(all_input_ids)  

    embeddings.append(torch.mean(features[0][0], dim=0))

stacked_tensor = torch.stack(embeddings)

X_test = torch.FloatTensor(stacked_tensor)
y_test = torch.FloatTensor(labels)


with open('dataset_mean_test.pkl','wb') as f:
    cPickle.dump((X_test, y_test),f)

'''    

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

# Train and test set generation 
dr_tr = DataReader('./Data/olid-training-v1.tsv','A')
data_tr, labels_tr = dr_tr.get_labelled_data()
dr_tst = DataReader('./Data/testset-levela.tsv','A')
data_tst,label_tst = dr_tst.get_test_data()

data_tr = data_tr[:]
labels_tr = labels_tr[:]

data_tst = data_tst[:]
labels_tst = label_tst[:]

lines_tr = [normalizeTweet(line) for line in data_tr]
lines_tst = [normalizeTweet(line) for line in data_tst]

embeddings_tr = []
embeddings_tst = []

# Generating embeddings
for line in tqdm(lines_tr,'Generating train embeddings'):
    subword = '<s> ' + bpe.encode(line) + ' </s>'
    input_id = vocab.encode_line(subword, append_eos=False, add_if_not_exist=False).long().tolist()
    all_input_ids = torch.tensor([input_id], dtype=torch.long) 
    with torch.no_grad():  
        features = BERTweet(all_input_ids)  
    embeddings_tr.append(torch.mean(features[0][0], dim=0))

for line in tqdm(lines_tst,'Generating test embeddings'):
    subword = '<s> ' + bpe.encode(line) + ' </s>'
    input_id = vocab.encode_line(subword, append_eos=False, add_if_not_exist=False).long().tolist()
    all_input_ids = torch.tensor([input_id], dtype=torch.long) 
    with torch.no_grad():  
        features = BERTweet(all_input_ids)  
    embeddings_tst.append(torch.mean(features[0][0], dim=0))

stacked_tensor_tr = torch.stack(embeddings_tr)
stacked_tensor_tst = torch.stack(embeddings_tst)

X_train = torch.FloatTensor(stacked_tensor_tr)
y_train = torch.FloatTensor(labels_tr)
X_test = torch.FloatTensor(stacked_tensor_tst)
y_test = torch.FloatTensor(labels_tst)

# Dumping to pickle
with open('dataset_mean_train.pkl','wb') as f:
    cPickle.dump((X_train, y_train),f)

with open('dataset_mean_test.pkl','wb') as f:
    cPickle.dump((X_test, y_test),f)

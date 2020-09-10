from DataReader import DataReader
from TweetNormalizer import normalizeTweet
import torch
import tqdm
import _pickle as cPickle
from pprint import pprint
import spacy
import wikipedia
from wikipedia2vec import Wikipedia2Vec
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
nlp = spacy.load("en_core_web_sm")

# Loading data
dr_tr = DataReader('./Data/olid-training-v1.0.tsv','A')
tr_data, tr_labels = dr_tr.get_labelled_data()
dr_tst = DataReader('./Data/testset-levela.tsv','A')
tst_data, tst_label = dr_tst.get_test_data()

tst_data = tst_data[:]
tst_labels = tst_label[:]
tr_data = tr_data[:]
tr_labels = tr_label[:]

tr_entities = []
tst_entities = []

# Generating Noun-Phrase chunks using SpaCy
for line in tqdm.tqdm(tr_data):
    temp = []
    doc = nlp(normalizeTweet(line))
    for chunk in doc.noun_chunks:
        if chunk.text!="@USER" and chunk.text!="HTTPURL" and chunk.text not in stop_words:
            entities = wikipedia.search(chunk.text,results=2)
            for entity in entities:
                temp.append(entity)
    tr_entities.append(temp)

for line in tqdm.tqdm(tst_data):
    temp = []
    doc = nlp(normalizeTweet(line))
    for chunk in doc.noun_chunks:
        if chunk.text!="@USER" and chunk.text!="HTTPURL" and chunk.text not in stop_words:
            entities = wikipedia.search(chunk.text,results=2)
            for entity in entities:
                temp.append(entity)
    tst_entities.append(temp)

# Saving to files
with open('./pickles/spacy_train.pkl','wb') as f:
    cPickle.dump(tr_entities,f)
with open('./pickles/spacy_test.pkl','wb') as f:
    cPickle.dump(tst_entities,f)
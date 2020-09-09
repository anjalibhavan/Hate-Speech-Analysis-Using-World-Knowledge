from DataReader import DataReader
from TweetNormalizer import normalizeTweet
import tqdm
import _pickle as cPickle
from pprint import pprint
import stanza
import wikipedia
import nltk
from nltk.corpus import stopwords
from wikipedia2vec import Wikipedia2Vec
import os
from stanza.server import CoreNLPClient

os.environ['CORENLP_HOME']='/data/users/abhavan/stanford-corenlp-4.0.0'

def noun_phrases(_client, _text, _annotators=None):
    pattern = 'NP'
    matches = _client.tregex(_text,pattern,annotators=_annotators)
    #print(matches['sentences'])
    #for sent in (["\t"+sentence[match_id]['spanString'] for sentence in matches['sentences'] for match_id in sentence]):
    #    print(wikipedia.search(sent))
    print("\n".join(["\t"+sentence[match_id]['spanString'] for sentence in matches['sentences'] for match_id in sentence]))

dr = DataReader('./Data/olid-training-v1.0.tsv','A')
data,labels = dr.get_labelled_data()
data = data[:]

with CoreNLPClient(timeout=30000, memory='16G') as client:
    englishText = "She should ask a few native Americans what their take on this is."
    print('---')
    print(englishText)
    noun_phrases(client,englishText,_annotators="tokenize,ssplit,pos,lemma,parse")


#wiki2vec = Wikipedia2Vec.load('/data/users/abhavan/enwiki_20180420_100d.pkl')
#print(wiki2vec.get_entity_vector(annotats[0][0]))


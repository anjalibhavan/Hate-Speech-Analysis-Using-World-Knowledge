from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
from gensim.test.utils import get_tmpfile
from datetime import datetime
import multiprocessing
import _pickle as cPickle
print(datetime.now())

#wiki = WikiCorpus("enwiki-20200701-pages-articles.xml.bz2")
print('here1')
class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c for c in content], [title])
print('here2')

'''
print('here')
documents = TaggedWikiDocument(wiki)
print('saving to pickle')
with open('docs.pkl','wb') as f:
    cPickle.dump(documents,f)

print('here3')
cores = multiprocessing.cpu_count()
print(cores)
'''
f = open('docs.pkl','rb')
documents = cPickle.load(f)
'''
models = [
    Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=19, iter=10, workers=cores),
    Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19, iter =10, workers=cores)
]
print('build vocab')
print(datetime.now())

models[0].build_vocab(documents)
print(str(models[0]))

print(datetime.now())

models[1].reset_from(models[0])
print(str(models[1]))

print(datetime.now())

with open('beforetr.pkl', 'wb') as f:
    cPickle.dump(models,f)
'''
f = open('beforetr.pkl','rb')

models = cPickle.load(f)
print(datetime.now())

print('training models finally')
for i, model in enumerate(models):
    print(datetime.now())
    model.train(documents, total_examples = model.corpus_count, epochs = model.epochs)
    if i==0:
        fname = get_tmpfile("resdbow.model")
        model.save(fname)
        with open('dbow.pkl','wb') as f:
            cPickle.dump(model,f)
        print('done')
    else:
        fname = get_tmpfile("resdm.model")
        model.save(fname)
        with open('dm.pkl','wb') as f:
            cPickle.dump(model,f)
        print('done')




#model = Doc2Vec.load(fname)  





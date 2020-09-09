from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import multiprocessing
import _pickle as cPickle

wiki = WikiCorpus("enwiki-20200701-pages-articles.xml.bz2")

class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c for c in content], [title])

Documents = TaggedWikiDocument(wiki)

cores = multiprocessing.cpu_count()

models = [
    Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=19, iter=10, workers=cores),
    Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19, iter =10, workers=cores)
]

models[0].build_vocab(documents)
models[1].reset_from(models[0])

print('training models')
for i, model in enumerate(models):
    model.train(documents, total_examples = model.corpus_count, epochs = model.epochs)
    if i is 0:
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

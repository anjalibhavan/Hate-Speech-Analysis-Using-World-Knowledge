import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import _pickle as cPickle
from sklearn.neural_network import MLPClassifier
import torch
import csv
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, normalize
from DataReader import DataReader
from datetime import datetime
from sklearn.metrics import *
import itertools
import random
random.seed(3)

with open('final.txt','r') as f, open('fff.csv','r') as g, open('final.csv','w') as h:
    t = f.readlines()
    greader = csv.reader(g)
    hwriter = csv.writer(h)
    #i = 0
    for i, (tx,cs) in tqdm(enumerate(zip(t,greader))):
        if i<86:
            temp = cs
            line = tx.split()
            temp.append(line[2])
            hwriter.writerow(temp)
        else:
            hwriter.writerow(cs)
        



        
        
'''
print(datetime.now())

n_gram_min, n_gram_max = 2, 3 
                                                                
f = open('./pickles/dataset_mean_train.pkl', 'rb')
X_train, y_train = cPickle.load(f)
f = open('./pickles/caa_mean.pkl', 'rb')
X_test, y_test = cPickle.load(f)
f = open('./pickles/dmfeatures_train.pkl', 'rb')
X_etrain = cPickle.load(f)
f = open('./pickles/caa_embed.pkl', 'rb')
X_etest = cPickle.load(f)

dr = DataReader('./filtered.csv','A')
data,labels = dr.get_labelled_data()
data = data[:]

X_train = X_train.numpy()
X_test = X_test.numpy()
X_train = np.concatenate((X_train,X_etrain),axis=1)
X_test = np.concatenate((X_test,X_etest),axis=1)

print(X_test.shape)
clf = LinearSVC(max_iter = 10000)
y_test = torch.FloatTensor(y_test)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#y_pred = y_pred.numpy()
y_test = y_test.numpy()

print(y_pred[0])
print(y_test[0])
with open('fff.csv','w') as f:
    fwriter = csv.writer(f)
    for i in range(len(y_pred)):
        fwriter.writerow([str(int(y_pred[i])), str(int(y_test[i]))])


print("Accuracy", accuracy_score(y_pred,y_test))
print("F1", f1_score(y_pred,y_test))
print("Precision",precision_score(y_pred,y_test))
print("Recall",recall_score(y_pred,y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_pred,y_test))

'''
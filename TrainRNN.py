import numpy as np
import torch
import _pickle as cPickle
import sys
import getopt
from torchsummary import summary
from sklearn.metrics import *
from Feedforward import Feedforward

# Loading data
f = open('./pickles/dataset_mean_train.pkl', 'rb')
X_train, y_train = cPickle.load(f)
f = open('./pickles/dataset_mean_test.pkl', 'rb')
X_test, y_test = cPickle.load(f)
f = open('./pickles/wiki_train.pkl', 'rb')
X_etrain = cPickle.load(f)
f = open('./pickles/wiki_test.pkl', 'rb')
X_etest = cPickle.load(f)

# Concatenating BERT and entity embeddings
X_train = np.concatenate((X_train,X_etrain),axis=1)
X_test = np.concatenate((X_test,X_etest),axis=1)

y_train = torch.nn.functional.one_hot(y_train.to(torch.int64), num_classes=2)
y_test = torch.nn.functional.one_hot(y_test.to(torch.int64), num_classes=2)
y_train = y_train.float()
y_test = y_test.float()

# Setting model parameters
learning_rate = 1e-4
epochs = 10

# Initialize model and loss function
model = Feedforward(X_train.shape[1], X_train.shape[0])
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Print model summary
summary(model, input_size=(X_train.shape[1], X_train.shape[0]))

# Model loss evaluation before training
model.eval()
y_pred = model(X_test)
before_train = criterion(y_pred, y_test)
print('Test loss before training' , before_train.item())

# Model training
model.train()
for epoch in range(epochs):   

    optimizer.zero_grad()   
    y_pred = model(X_train)    
    loss = criterion(y_pred, y_train)
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    
    loss.backward()
    optimizer.step()

# Model loss evaluation after training
model.eval()
y_pred = model(X_test)
after_train = criterion(y_pred, y_test) 
print('Test loss after Training' , after_train.item())

y_pred = y_pred.detach().numpy()
y_test = y_test.numpy()
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)

print("Accuracy", accuracy_score(y_test,y_pred))
print("F1", f1_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))
print("Recall",recall_score(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
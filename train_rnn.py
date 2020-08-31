import numpy as np
import torch
import _pickle as cPickle
import sys
import getopt
from torchsummary import summary
from sklearn.metrics import *
import random
random.seed(3)

f = open('./pickles/dataset_mean_train.pkl', 'rb')
X_train, y_train = cPickle.load(f)
f = open('./pickles/dataset_mean_test.pkl', 'rb')
X_test, y_test = cPickle.load(f)

class Feedforward(torch.nn.Module):

        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.dp1 = torch.nn.Dropout(p = 0.5)
            self.relu1 = torch.nn.ReLU()
            
            self.rnn1 = torch.nn.RNNCell(self.hidden_size, self.hidden_size)
            self.tanh1 = torch.nn.Tanh()
            self.rnn2 = torch.nn.RNNCell(self.hidden_size, self.hidden_size)
            self.tanh2 = torch.nn.Tanh()
            self.rnn3 = torch.nn.RNNCell(self.hidden_size, self.hidden_size)
            self.tanh3 = torch.nn.Tanh()
            self.dp2 = torch.nn.Dropout(p = 0.5)

            self.fc2 = torch.nn.Linear(self.hidden_size, 2)
            self.sigmoid = torch.nn.Sigmoid()        
        
        def forward(self, x):
            fc1 = self.fc1(x)
            dp1 = self.dp1(fc1)
            relu1 = self.relu1(dp1)
            
            rnn1 = self.rnn1(relu1)
            tanh1 = self.tanh1(rnn1)
            rnn2 = self.rnn2(tanh1)
            tanh2 = self.tanh2(rnn2)
            rnn3 = self.rnn3(tanh2)
            tanh3 = self.tanh3(rnn3)
            dp2 = self.dp2(tanh3)

            output = self.fc2(dp2)
            output = self.sigmoid(output)
            return output

y_train = torch.nn.functional.one_hot(y_train.to(torch.int64), num_classes=2)
y_test = torch.nn.functional.one_hot(y_test.to(torch.int64), num_classes=2)
y_train = y_train.float()
y_test = y_test.float()

learning_rate = 1e-4
argv = sys.argv[1:]
epochs = int(argv[0])

model = Feedforward(X_train.shape[1], X_train.shape[0])
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

summary(model, input_size=(X_train.shape[1], X_train.shape[0]))

model.eval()
y_pred = model(X_test)
before_train = criterion(y_pred, y_test)
print('Test loss before training' , before_train.item())

model.train()
for epoch in range(epochs):   

    optimizer.zero_grad()   
    y_pred = model(X_train)    
    loss = criterion(y_pred, y_train)
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    
    loss.backward()
    optimizer.step()

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
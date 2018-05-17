import torch
import math
import numpy as np
import scipy.io as sci
import torch.nn as nn
import os
import h5py
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch.utils.serialization import load_lua
#torch.cuda.set_device(1)

N, T ,D= 100, 10, 200	#opt.batch_size, opt.seq_length , word_dim	

###############################################################################
class TrainDataset(Data.Dataset):
    def __init__(self):
        a=np.array(h5py.File('/home/lu/code/pytorch/data_dir/CCDD10.mat')['traindata'])
        #b=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data11.mat')['train_data1'])
        self.data = np.transpose(a)

    def __getitem__(self, idx):
        
        data_ori = torch.from_numpy(self.data[idx])
        data = data_ori[0:2000]*0.0048
        label = data_ori[2001]
        return data, label

    def __len__(self):
        return len(self.data)

trainset = TrainDataset()
train_loader = Data.DataLoader(trainset, batch_size = N,shuffle = True)

################################################################################
class TestDataset(Data.Dataset):
    def __init__(self):
        a=np.array(h5py.File('/home/lu/code/pytorch/data_dir/CCDD10.mat')['testdata'])
        #b=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data11.mat')['test_data1'])
        self.data = np.transpose(a)

    def __getitem__(self, idx):
        
        data_ori = torch.from_numpy(self.data[idx])
        data = data_ori[0:2000]*0.0048
        label = data_ori[2001]
        return data, label

    def __len__(self):
        return len(self.data)

testset = TestDataset()
test_loader = Data.DataLoader(testset, batch_size = 1, shuffle = True)

##################################################################
# build a nerul network with nn.RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_szie1, hidden_szie2, output_size):
        super(RNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_szie1, 4, dropout = 0.3)
        self.r2h = nn.Linear(hidden_szie1, hidden_szie2)
        self.h2o = nn.Linear(hidden_szie2, output_size)

    def forward(self, input):
        hidden, _ = self.rnn(input)

        fc1 = F.relu(self.r2h(hidden[T-1]))
        output = self.h2o(fc1)

        return output

class RNN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=4, dropout = 0.3, max_length = T):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.dropout = dropout
        self.output_size = output_size

        self.gru = nn.GRU(input_size, hidden_size, self.n_layers, dropout = 0.3, bidirectional = True)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        
        output, hidden = self.gru(input)
        output = output.transpose(0,1)
        hidden = torch.cat((hidden[0],hidden[1]),1)

        attn_weights = self.attn(hidden)
        attn_weights = torch.bmm(attn_weights.unsqueeze(1), output)
        attn_weights = F.softmax(self.attn(attn_weights.squeeze(1)))

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), output)
        attn_applied = attn_applied.squeeze(1)
        output = self.attn_combine(attn_applied)
        output = F.relu(output)
        output = self.out(output)
        #output = F.softmax(output)
        #print(output)
        return output

##################################################################
# train loop
model = RNN(D, 400, 200, 2)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(input, label):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, label)
    loss.backward()

    optimizer.step()
    return output, loss.data[0]

def test(input):

    output = model(input)
    top_n, top_i = output.data.topk(1)
    
    return top_i[0][0]

##################################################################
# let's train it

n_epochs = 30
current_loss = 0
all_losses = []
err_rate = []
err = 0
confusion = torch.zeros(2,2)

for epoch in range(1, n_epochs+1):
    for step1,(batch_x, batch_y) in enumerate(train_loader):
        batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
        batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))

        output, loss = train(batch_x.view(N,T,D).transpose(0,1), batch_y)
        current_loss += loss

    for step2,(test_x, test_y) in enumerate(test_loader):
        
        test_x = Variable(test_x.type('torch.cuda.FloatTensor'))
        test_y = test_y.type('torch.cuda.LongTensor')

        guess = test(test_x.view(1,T,-1).transpose(0,1))
        confusion[guess][test_y[0]] += 1

    sen = (confusion[0][0])/((confusion[0][0]+confusion[0][1]))
    acc = (confusion[0][0]+confusion[1][1])/step2
    
    all_losses.append(current_loss / step1)
    err_rate.append(acc*100)
    
    current_loss = 0
    print('%d epoch: acc = %.2f, sen = %.2f%%'%(epoch, acc*100, sen*100))
    err = 0   
    confusion = torch.zeros(2,2)

#print(confusion)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


plt.figure()
plt.plot(all_losses)
plt.title('loss')
plt.figure()
plt.plot(err_rate)
plt.title('err')

#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(confusion.numpy())
#fig.colorbar(cax)

plt.show()


RNN_loss = open('CCDD10.csv', 'w+')
for item in range(len(all_losses)):
    RNN_loss.write(str(all_losses[item]) + '\n')

RNN_loss.close()
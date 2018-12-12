###multi rnn
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

#torch.cuda.set_device(1)
use_cuda = torch.cuda.is_available()

N, T ,D ,L, O= 100, 10, 200 ,8 ,2	#batch_size, seq_length , word_dim	,leads
hidden_size = 50

###############################################################################
class TrainDataset(Data.Dataset):
    def __init__(self):
        a=np.array(h5py.File('/home/lu/code/pytorch/data_dir/traindata.mat')['traindata'])
        #b=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data11.mat')['train_data1'])
        self.data = np.transpose(a)

    def __getitem__(self, idx):
        
        data_ori = torch.from_numpy(self.data[idx])
        data = data_ori[0:16000]
        label = data_ori[16000]
        return data, label

    def __len__(self):
        return len(self.data)

trainset = TrainDataset()
train_loader = Data.DataLoader(trainset, batch_size = N,shuffle = True)

################################################################################
class TestDataset(Data.Dataset):
    def __init__(self):
        a=np.array(h5py.File('/home/lu/code/pytorch/data_dir/testdata.mat')['testdata'])
        #b=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data11.mat')['test_data1'])
        self.data = np.transpose(a)

    def __getitem__(self, idx):
        
        data_ori = torch.from_numpy(self.data[idx])
        data = data_ori[0:16000]
        label = data_ori[16000]
        return data, label

    def __len__(self):
        return len(self.data)

testset = TestDataset()
test_loader = Data.DataLoader(testset, batch_size = 1, shuffle = True)

##################################################
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

        #print(output)
        return output

class RNN(nn.Module):
    def __init__(self, input_size, hidden_szie1, hidden_szie2, output_size):
        super(RNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_szie1, 2, dropout = 0.3)
        self.r2h = nn.Linear(hidden_szie1, hidden_szie2)
        self.h2o = nn.Linear(hidden_szie2, output_size)

    def forward(self, input):
        hidden, _ = self.rnn(input)
        fc1 = F.relu(self.r2h(hidden[T-1]))
        output = self.h2o(fc1)
        #output = F.softmax(output)

        return output

class MLP(nn.Module):
    def __init__(self, input_size, hidden_szie, output_size):
        super(MLP, self).__init__()
        
        self.i2h = nn.Linear(input_size, hidden_szie)
        self.h2o = nn.Linear(hidden_szie, output_size)

    def forward(self, input):
        hidden = F.relu(self.i2h(input))
        output = self.h2o(hidden)

        return output

n_epochs = 20
current_loss = 0
all_losses = []
err_rate = []
confusion = torch.zeros(6, 6)
err = 0
learning_rate = 0.001

rnn1 = RNN(200, 200, 100, 20)
rnn2 = RNN(200, 200, 100, 20)
rnn3 = RNN(200, 200, 100, 20)
rnn4 = RNN(200, 200, 100, 20)
rnn5 = RNN(200, 200, 100, 20)
rnn6 = RNN(200, 200, 100, 20)
rnn7 = RNN(200, 200, 100, 20)
rnn8 = RNN(200, 200, 100, 20)
mlp = MLP(160,100,2)

if use_cuda:
    rnn1 = rnn1.cuda()
    rnn2 = rnn2.cuda()
    rnn3 = rnn3.cuda()
    rnn4 = rnn4.cuda()
    rnn5 = rnn5.cuda()
    rnn6 = rnn6.cuda()
    rnn7 = rnn7.cuda()
    rnn8 = rnn8.cuda()
    mlp = mlp.cuda()

rnn1_optimizer = torch.optim.Adam(rnn1.parameters(), lr=learning_rate)
rnn2_optimizer = torch.optim.Adam(rnn2.parameters(), lr=learning_rate)
rnn3_optimizer = torch.optim.Adam(rnn3.parameters(), lr=learning_rate)
rnn4_optimizer = torch.optim.Adam(rnn4.parameters(), lr=learning_rate)
rnn5_optimizer = torch.optim.Adam(rnn5.parameters(), lr=learning_rate)
rnn6_optimizer = torch.optim.Adam(rnn6.parameters(), lr=learning_rate)
rnn7_optimizer = torch.optim.Adam(rnn7.parameters(), lr=learning_rate)
rnn8_optimizer = torch.optim.Adam(rnn8.parameters(), lr=learning_rate)
mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

for epoch in range(1, n_epochs+1):

    for step1,(batch_x, batch_y) in enumerate(train_loader):

        batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
        batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))

        x = batch_x.view(N,L,-1).transpose(0,1).clone()

        rnn1_optimizer.zero_grad()
        rnn2_optimizer.zero_grad()
        rnn3_optimizer.zero_grad()
        rnn4_optimizer.zero_grad()
        rnn5_optimizer.zero_grad()
        rnn6_optimizer.zero_grad()
        rnn7_optimizer.zero_grad()
        rnn8_optimizer.zero_grad()
        mlp_optimizer.zero_grad()

        loss = 0

        out1 = rnn1(x[0].view(N,T,D).transpose(0,1))
        out2 = rnn2(x[1].view(N,T,D).transpose(0,1))
        out3 = rnn3(x[2].view(N,T,D).transpose(0,1))
        out4 = rnn4(x[3].view(N,T,D).transpose(0,1))
        out5 = rnn5(x[4].view(N,T,D).transpose(0,1))
        out6 = rnn6(x[5].view(N,T,D).transpose(0,1))
        out7 = rnn7(x[6].view(N,T,D).transpose(0,1))
        out8 = rnn8(x[7].view(N,T,D).transpose(0,1))
        #print(out1)
        rnn_out = torch.cat((out1,out2,out3,out4,out5,out6,out7,out8),1)

        output = mlp(rnn_out)
        loss = criterion(output, batch_y)

        loss.backward()

        rnn1_optimizer.step()
        rnn2_optimizer.step()
        rnn3_optimizer.step()
        rnn4_optimizer.step()
        rnn5_optimizer.step()
        rnn6_optimizer.step()
        rnn7_optimizer.step()
        rnn8_optimizer.step()
        mlp_optimizer.step()

        current_loss += loss.data[0]

    for step2,(test_x, test_y) in enumerate(test_loader):

        test_x = Variable(test_x.type('torch.cuda.FloatTensor'))
        test_y = test_y.type('torch.cuda.LongTensor')
        
        x = test_x.view(1,L,-1).transpose(0,1).clone()
        out1 = rnn1(x[0].view(1,T,D).transpose(0,1))
        out2 = rnn2(x[1].view(1,T,D).transpose(0,1))
        out3 = rnn3(x[2].view(1,T,D).transpose(0,1))
        out4 = rnn4(x[3].view(1,T,D).transpose(0,1))
        out5 = rnn5(x[4].view(1,T,D).transpose(0,1))
        out6 = rnn6(x[5].view(1,T,D).transpose(0,1))
        out7 = rnn7(x[6].view(1,T,D).transpose(0,1))
        out8 = rnn8(x[7].view(1,T,D).transpose(0,1))

        rnn_out = torch.cat((out1,out2,out3,out4,out5,out6,out7,out8),1)

        output = mlp(rnn_out)
        top_n, top_i = output.data.topk(1)
        confusion[top_i[0][0]][test_y[0]] += 1

    sen = (confusion[0][0])/((confusion[0][0]+confusion[0][1]))
    acc = (confusion[0][0]+confusion[1][1])/step2
    
    all_losses.append(current_loss / step1)
    err_rate.append(acc*100)
    
    print('%d epoch: acc = %.2f, sen = %.2f, loss = %f%%'%(epoch, acc*100, sen*100, current_loss/step1))
    current_loss = 0
    err = 0   
    confusion = torch.zeros(2,2)

plt.figure()
plt.plot(all_losses)
plt.title('loss')
plt.figure()
plt.plot(err_rate)
plt.title('err')

print(confusion)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

plt.show()

RNN_loss = open('Mul-RNN-L-20.csv', 'w+')
for item in range(len(all_losses)):
    RNN_loss.write(str(all_losses[item]) + '\n')

RNN_loss.close()
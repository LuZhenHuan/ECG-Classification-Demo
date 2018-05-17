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
import random

torch.cuda.set_device(1)
use_cuda = torch.cuda.is_available()

N, T ,D ,L, O= 100, 10, 50 ,8 ,2	#batch_size, seq_length , word_dim	,leads
hidden_size = 50

class CNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.001, max_length=L):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.dropout = nn.Dropout(self.dropout_p)

        self.conv1 = nn.Conv1d(L,8,100)
        self.conv2 = nn.Conv1d(8,16,100)
        self.conv3 = nn.Conv1d(16,16,100)
        #self.conv4 = nn.Conv1d(3,1,10)

        self.fc1 = nn.Linear(2608, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_outputs):


        cnn_out = F.max_pool1d(F.relu(self.conv1(encoder_outputs)),2)
        c1 = cnn_out
        cnn_out = F.max_pool1d(F.relu(self.conv2(cnn_out)),2)
        c2 = cnn_out
        cnn_out = F.max_pool1d(F.relu(self.conv3(cnn_out)),2)
        c3 = cnn_out
        #cnn_out = F.max_pool1d(F.sigmoid(self.conv4(cnn_out)),2)
        #print(cnn_out)
        cnn_out = cnn_out.view(-1,2608)
        output = F.relu(self.fc1(cnn_out))
        output = self.out(output)
        
        return output, c1, c2 ,c3

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout = 0.3, max_length = T):
        super(Attention, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.output_size = output_size
        self.dropout = dropout

        self.gru = nn.GRU(input_size, hidden_size, self.n_layers, dropout = 0.3,bidirectional = True)
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
        
        return output, hidden, attn_weights

cnn = torch.load('/home/lu/code/pytorch/data_dir/cnn.pkl').cuda()
rnn = torch.load('/home/lu/code/pytorch/data_dir/rnn.pkl').cuda()

def test(input_variable, cnn, rnn):
    
    cnn_output, c1, c2, c3 = cnn(input_variable)
    cnn_output = cnn_output.view(1, T, D).transpose(0,1)
    rnn_output, hidden, attn_weight = rnn(cnn_output)
    top_n, top_i = rnn_output.data.topk(1)

    return top_i[0][0], c1, c2, c3, attn_weight

#data = torch.from_numpy(np.transpose(np.array(h5py.File('/home/lu/code/pytorch/data_dir/data1_3.mat')['testdata'])))   

data = torch.from_numpy(np.transpose(np.array(h5py.File('/home/lu/code/pytorch/data_dir/testdata.mat')['testdata'])))


for i in range(10):
    j = random.randint(10000, 12800)
    print(j)
    if data[j][16000] == 1:
        
        test_x = data[j][0:16000]*0.0048
        print(test_x)

        test_x = Variable(test_x.type('torch.cuda.FloatTensor'))

        guess, c1, c2, c3, attn_weight = test(test_x.view(1,L,-1), cnn, rnn)

        top_n, top_i = attn_weight.data.topk(4)
        num1 = top_i[0][0]
        num2 = top_i[0][1]
        num3 = top_i[0][2]
        num4 = top_i[0][3]

        print(guess, num1, num2, attn_weight)
        a = test_x.data.type('torch.FloatTensor').numpy()
        a = a[2000:4000]
        plt.figure()
        plt.title(str(guess)+','+str(num1)+','+str(num2))
        plt.plot(a)
        p = plt.axvspan(num1*200,(num1+1)*200,facecolor = 'b',alpha = 0.7)
        p = plt.axvspan(num2*200,(num2+1)*200,facecolor = 'b',alpha = 0.5)
        p = plt.axvspan(num3*200,(num3+1)*200,facecolor = 'b',alpha = 0.3)
        p = plt.axvspan(num4*200,(num4+1)*200,facecolor = 'b',alpha = 0.1)
    
plt.show()


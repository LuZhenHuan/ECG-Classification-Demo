import torch
import math
import numpy as np
import scipy.io as sci
import torch.nn as nn
import h5py
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker   
import random

N, T ,D= 100, 80, 300	#opt.batch_size, opt.seq_length , word_dim	
 
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=3,max_length = T):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
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
        #output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(attn_applied)
        output = F.relu(output)
        output = self.out(output)
        #output = F.log_softmax(self.out(output[0]))
        
        return output, hidden, attn_weights

encoder = torch.load('/home/lu/code/pytorch/data_dir/encoder.pkl').cuda()
#decoder = torch.load('/home/lu/code/pytorch/data_dir/decoder.pkl').cuda()

def test(input_variable, encoder):

    decoder_output, decoder_hidden, attn_weights = encoder(input_variable)
    
    top_n, top_i = decoder_output.data.topk(1)
    
    return top_i[0][0], attn_weights

data = torch.from_numpy(np.transpose(np.array(h5py.File('/home/lu/code/pytorch/data_dir/data1_3.mat')['testdata'])))   

for i in range(10):
    j = random.randint(0, 2800)
    print(j)
    if data[j][24001] == 1:
        test_x = data[j][0:24000]

        test_x = Variable(test_x.type('torch.cuda.FloatTensor'))
        guess, attn_weight = test(test_x.view(1,T,-1).transpose(0,1), encoder)

        top_n, top_i = attn_weight.data.topk(2)
        num1 = top_i[0][0]
        num2 = top_i[0][1]
        print(guess, num1, num2, attn_weight)
        a = test_x.data.type('torch.FloatTensor').numpy()

        plt.figure()
        plt.title(str(guess)+','+str(num1)+','+str(num2))
        plt.plot(a)
        p = plt.axvspan(num1*D,(num1+1)*D,facecolor = 'r',alpha = 0.4)
        p = plt.axvspan(num2*D,(num2+1)*D,facecolor = 'r',alpha = 0.4)
    
plt.show()


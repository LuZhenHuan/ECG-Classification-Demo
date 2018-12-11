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

torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()

N, T ,D ,L, O= 10, 8, 250 ,12 ,511	#batch_size, seq_length , word_dim	,leads

###############################################################################
class TrainDataset(Data.Dataset):
    def __init__(self):
        a=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data10.mat')['train_data0'])
        b=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data11.mat')['train_data1'])
        
        self.data = np.transpose(np.hstack((a,b)))

    def __getitem__(self, idx):
        
        data_ori = torch.from_numpy(self.data[idx])
        data = data_ori[0:24000]
        label = data_ori[24001]
        
        return data, label

    def __len__(self):
        return len(self.data)

trainset = TrainDataset()
train_loader = Data.DataLoader(trainset, batch_size = N,shuffle = True)

################################################################################
class TestDataset(Data.Dataset):
    def __init__(self):
        a=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data10.mat')['test_data0'])
        b=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data11.mat')['test_data1'])
        
        self.data = np.transpose(np.hstack((a,b)))

    def __getitem__(self, idx):
        
        data_ori = torch.from_numpy(self.data[idx])
        data = data_ori[0:24000]
        label = data_ori[24001]
        
        return data, label

    def __len__(self):
        return len(self.data)

testset = TestDataset()
test_loader = Data.DataLoader(testset, batch_size = 1, shuffle = True)

##################################################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout = 0.3, max_length = T):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.output_size = output_size
        self.dropout = dropout

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
        
        return output, hidden, attn_weights

class CNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=L):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.cnn = nn.Conv2d(1,1,12,5)
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc1 = nn.Linear(100, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, hidden, encoder_outputs):

        encoder_outputs = encoder_outputs.unsqueeze(1)
        cnn_out = F.relu(self.cnn(encoder_outputs))
        cnn_out = cnn_out.view(-1,100)
        output = F.relu(self.fc1(cnn_out))
        output = self.out(output)
        
        return output, hidden

def train(input_variable, target_variable, encoder1, encoder2, encoder3, encoder4, encoder5,
        encoder6, encoder7, encoder8, encoder9, encoder10, encoder11, encoder12, cnn, 
        encoder1_optimizer, encoder2_optimizer,encoder3_optimizer,encoder4_optimizer,
        encoder5_optimizer,encoder6_optimizer,encoder7_optimizer,encoder8_optimizer,
        encoder9_optimizer,encoder10_optimizer,encoder11_optimizer,encoder12_optimizer,
        cnn_optimizer, criterion, max_length=T):

    encoder1_optimizer.zero_grad()
    encoder2_optimizer.zero_grad()
    encoder3_optimizer.zero_grad()
    encoder4_optimizer.zero_grad()
    encoder5_optimizer.zero_grad()
    encoder6_optimizer.zero_grad()
    encoder7_optimizer.zero_grad()
    encoder8_optimizer.zero_grad()
    encoder9_optimizer.zero_grad()
    encoder10_optimizer.zero_grad()
    encoder11_optimizer.zero_grad()
    encoder12_optimizer.zero_grad()
    cnn_optimizer.zero_grad()

    loss = 0
    
    lead_output = Variable(torch.Tensor().type('torch.cuda.FloatTensor'))
    lead_hidden = Variable(torch.Tensor().type('torch.cuda.FloatTensor'))

    input_part = input_variable[0].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder12(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))

    input_part = input_variable[1].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder1(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))

    input_part = input_variable[2].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder2(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))

    input_part = input_variable[3].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder3(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[4].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder4(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[5].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder5(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[6].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder6(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[7].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder7(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[8].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder8(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[9].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder9(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[10].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder10(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[11].clone()
    input_lead = input_part.view(N,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder11(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    lead_output = lead_output.view(N,L,O)
    model_output, model_hidden = cnn(lead_hidden, lead_output)
   
    loss = criterion(model_output, target_variable)

    loss.backward()

    encoder1_optimizer.step()
    encoder2_optimizer.step()
    encoder3_optimizer.step()
    encoder4_optimizer.step()
    encoder5_optimizer.step()
    encoder6_optimizer.step()
    encoder7_optimizer.step()
    encoder8_optimizer.step()
    encoder9_optimizer.step()
    encoder10_optimizer.step()
    encoder11_optimizer.step()
    encoder12_optimizer.step()

    cnn_optimizer.step()

    return loss.data[0]

def test(input_variable, encoder1, encoder2, encoder3, encoder4, encoder5,
        encoder6, encoder7, encoder8, encoder9, encoder10, encoder11, encoder12, cnn):
    
    lead_output = Variable(torch.Tensor().type('torch.cuda.FloatTensor'))
    lead_hidden = Variable(torch.Tensor().type('torch.cuda.FloatTensor'))

    input_part = input_variable[0].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder12(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))

    input_part = input_variable[1].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder1(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))

    input_part = input_variable[2].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder2(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))

    input_part = input_variable[3].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder3(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[4].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder4(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[5].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder5(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[6].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder6(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[7].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder7(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[8].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder8(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[9].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder9(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[10].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder10(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
    
    input_part = input_variable[11].clone()
    input_lead = input_part.view(1,T,D).transpose(0,1).clone()
    encoder_outputs, encoder_hidden, attn_weights = encoder11(input_lead)
    lead_output = Variable(torch.cat((lead_output.data, encoder_outputs.data),1))
   
    lead_output = lead_output.view(1,L,O)
    model_output, model_hidden = cnn(lead_hidden, lead_output)

    top_n, top_i = model_output.data.topk(1)
    return top_i[0][0]

def trainIters(encoder1, encoder2, encoder3, encoder4, encoder5,encoder6, encoder7, encoder8, 
        encoder9, encoder10, encoder11, encoder12, cnn, epoch=10, learning_rate=0.001):

    n_epochs = epoch
    current_loss = 0
    all_losses = []
    err_rate = []
    confusion = torch.zeros(6, 6)
    err = 0

    encoder1_optimizer = torch.optim.Adam(encoder1.parameters(), lr=learning_rate)
    encoder2_optimizer = torch.optim.Adam(encoder2.parameters(), lr=learning_rate)
    encoder3_optimizer = torch.optim.Adam(encoder3.parameters(), lr=learning_rate)
    encoder4_optimizer = torch.optim.Adam(encoder4.parameters(), lr=learning_rate)
    encoder5_optimizer = torch.optim.Adam(encoder5.parameters(), lr=learning_rate)
    encoder6_optimizer = torch.optim.Adam(encoder6.parameters(), lr=learning_rate)
    encoder7_optimizer = torch.optim.Adam(encoder7.parameters(), lr=learning_rate)
    encoder8_optimizer = torch.optim.Adam(encoder8.parameters(), lr=learning_rate)
    encoder9_optimizer = torch.optim.Adam(encoder9.parameters(), lr=learning_rate)
    encoder10_optimizer = torch.optim.Adam(encoder10.parameters(), lr=learning_rate)
    encoder11_optimizer = torch.optim.Adam(encoder11.parameters(), lr=learning_rate)
    encoder12_optimizer = torch.optim.Adam(encoder12.parameters(), lr=learning_rate)

    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs+1):
        for step1,(batch_x, batch_y) in enumerate(train_loader):

            batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
            batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))
            loss = train(batch_x.view(N,L,-1).transpose(0,1), batch_y, encoder1, encoder2, encoder3, encoder4, encoder5,
                    encoder6, encoder7, encoder8, encoder9, encoder10, encoder11, encoder12, cnn, 
                    encoder1_optimizer, encoder2_optimizer,encoder3_optimizer,encoder4_optimizer,
                    encoder5_optimizer,encoder6_optimizer,encoder7_optimizer,encoder8_optimizer,
                    encoder9_optimizer,encoder10_optimizer,encoder11_optimizer,encoder12_optimizer,
                    cnn_optimizer, criterion)
            current_loss += loss

        for step2,(test_x, test_y) in enumerate(test_loader):

            test_x = Variable(test_x.type('torch.cuda.FloatTensor'))
            test_y = test_y.type('torch.cuda.LongTensor')
            guess = test(test_x.view(1,L,-1).transpose(0,1), encoder1, encoder2, encoder3, encoder4, encoder5,
                        encoder6, encoder7, encoder8, encoder9, encoder10, encoder11, encoder12, cnn)
            #print('g',guess,'t',test_y[0])
            if guess != test_y[0]:
                    err += 1
        
            if epoch == n_epochs:
                confusion[guess][test_y[0]] += 1   

        print(current_loss/(step1+1))
        all_losses.append(current_loss/(step1+1))
        err_rate.append((1-err/step2)*100)
        print(err)
        print('%d epoch:, err number = %d, err rate = %.2f%%'%(epoch, err, ((1-err/step2)*100)))
    
        current_loss = 0
        err = 0

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

hidden_size = 200
encoder1 = EncoderRNN(D, hidden_size, O)
encoder2 = EncoderRNN(D, hidden_size, O)
encoder3 = EncoderRNN(D, hidden_size, O)
encoder4 = EncoderRNN(D, hidden_size, O)
encoder5 = EncoderRNN(D, hidden_size, O)
encoder6 = EncoderRNN(D, hidden_size, O)
encoder7 = EncoderRNN(D, hidden_size, O)
encoder8 = EncoderRNN(D, hidden_size, O)
encoder9 = EncoderRNN(D, hidden_size, O)
encoder10 = EncoderRNN(D, hidden_size, O)
encoder11 = EncoderRNN(D, hidden_size, O)
encoder12 = EncoderRNN(D, hidden_size, O)

cnn = CNN(50, 2)

if use_cuda:
    encoder1 = encoder1.cuda()
    encoder2 = encoder2.cuda()
    encoder3 = encoder3.cuda()
    encoder4 = encoder4.cuda()
    encoder5 = encoder5.cuda()
    encoder6 = encoder6.cuda()
    encoder7 = encoder7.cuda()
    encoder8 = encoder8.cuda()
    encode9r = encoder9.cuda()
    encoder10 = encoder10.cuda()
    encoder11 = encoder11.cuda()
    encoder12 = encoder12.cuda()

    cnn = cnn.cuda()

trainIters(encoder1, encoder2, encoder3, encoder4, encoder5,encoder6, encoder7, encoder8, 
        encoder9, encoder10, encoder11, encoder12, cnn, 10)

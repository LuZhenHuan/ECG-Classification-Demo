import torch
import h5py
import math
import numpy as np
import scipy.io as sci
import torch.nn as nn
import torch.nn.init as init
import os
from sklearn import preprocessing
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from PIL import Image
import torchvision.transforms as transforms

use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)

###############################################################################
class TrainDataset(Data.Dataset):
    def __init__(self):
        a=np.array(h5py.File('/home/lu/code/pytorch/data1_60.mat')['traina'])
        #b=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data11.mat')['train_data1'])
        self.data = np.transpose(a)
        
    def __getitem__(self, idx):
        
        data_ori1 = torch.from_numpy(self.data[idx])
        label = data_ori1[-1]-1

        where_are_nan = np.isnan(self.data[idx])
        self.data[idx][where_are_nan] = 0
        data1 = preprocessing.scale(self.data[idx])
        data_ori = torch.from_numpy(data1)
        data = data_ori[:153600]
        #print(data)
        #print(label)
        return data, label

    def __len__(self):
        return len(self.data)

trainset = TrainDataset()
train_loader = Data.DataLoader(trainset, batch_size = 72,shuffle = True)

################################################################################

class TestDataset(Data.Dataset):
    def __init__(self):
        a=np.array(h5py.File('/home/lu/code/pytorch/data1_60.mat')['trainjj'])
        #b=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data11.mat')['train_data1'])
        self.data = np.transpose(a)

    def __getitem__(self, idx):
        
        data_ori1 = torch.from_numpy(self.data[idx])
        label = data_ori1[-1]-1

        where_are_nan = np.isnan(self.data[idx])
        self.data[idx][where_are_nan] = 0
        data1 = preprocessing.scale(self.data[idx])
        data_ori = torch.from_numpy(data1)
        data = data_ori[:153600]
        #print(data.size())
        return data, label

    def __len__(self):
        return len(self.data)

testset = TestDataset()
test_loader = Data.DataLoader(testset, batch_size = 1, shuffle = True)

##################################################
class CNN(nn.Module):
    def __init__(self, hidden_size, output_size):
    #def __init__(self):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        #self.norm = nn.BatchNorm1d(60)

        self.conv1 = nn.Conv1d(60,30,100,stride = 2)
        self.conv2 = nn.Conv1d(30,16,100,stride = 2)
        #self.conv3 = nn.Conv1d(16,8,100,stride = 2)
        #self.conv4 = nn.Conv1d(3,1,10)


        self.fc1 = nn.Linear(80, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        


    def forward(self, x): 
        #x = init.normal(x)
        #cnn_out = self.norm(x)


        cnn_out = F.max_pool1d(F.tanh(self.conv1(x)),7)
        cnn_out = F.dropout(cnn_out,p = 0.25)
        cnn_out = F.max_pool1d(F.tanh(self.conv2(cnn_out)),7)
        cnn_out = F.dropout(cnn_out,p = 0.25)
        #cnn_out = F.max_pool1d(F.relu(self.conv3(cnn_out)),2)
        #cnn_out = F.max_pool1d(F.sigmoid(self.conv4(cnn_out)),2)
        #print(cnn_out)
        cnn_out = cnn_out.view(-1,80)
        output = F.leaky_relu(self.fc1(cnn_out))
        output = self.out(output)
        #print(output)
        return output


def train(input_variable, target_variable, cnn, cnn_optimizer, criterion):

    cnn_optimizer.zero_grad()

    loss = 0

    cnn_output = cnn(input_variable)

    loss = criterion(cnn_output, target_variable)

    loss.backward()

    cnn_optimizer.step()

    return loss.data[0]


def test(input_variable, cnn):
    
    cnn_output = cnn(input_variable)

    top_n, top_i = cnn_output.data.topk(1)

    return top_i[0][0]


def trainIters(cnn, epoch, learning_rate=0.0001):

    n_epochs = epoch
    current_loss = 0
    all_losses = []
    err_rate = []
    confusion = torch.zeros(4, 4)
    err = 0
    accTemp = 0

    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate,weight_decay = 0.1)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs+1):
        for step1,(batch_x, batch_y) in enumerate(train_loader):

            batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
            batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))

            loss = train(batch_x.view(72,2560,60).transpose(1,2), batch_y, cnn, cnn_optimizer, criterion)
            
            current_loss += loss


        for step2,(test_x, test_y) in enumerate(test_loader):

            test_x = Variable(test_x.type('torch.cuda.FloatTensor'))
            test_y = test_y.type('torch.cuda.LongTensor')

            guess = test(test_x.view(1,2560,60).transpose(1,2), cnn)
            #print(guess,test_y[0])
            confusion[guess][test_y[0]] += 1
            
            
            if step2 == 35:
                print(confusion)
                break
 

        sen = (confusion[0][0])/((confusion[0][0]+confusion[0][1]+1))
        acc = (confusion[0][0]+confusion[1][1]+confusion[2][2]+confusion[3][3])/(step2+1)
       
        all_losses.append(current_loss / step1)
        print(current_loss)
        #err_rate.append(acc*100)

        print('%d epoch: acc = %.2f, sen = %.2f%%'%(epoch, acc*100, sen*100))
        current_loss = 0
        err = 0   
        confusion = torch.zeros(4,4)

cnn = CNN(500,4)

if use_cuda:
    cnn = cnn.cuda()

trainIters(cnn, 30)

CNN_loss = open('temp.csv', 'w+')
for item in range(len(all_losses)):
    CNN_loss.write(str(all_losses[item]) + '\n')

CNN_loss.close()

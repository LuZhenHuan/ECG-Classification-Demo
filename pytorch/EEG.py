import torch
import math
import numpy as np
import scipy.io as sci
import torch.nn as nn
import torch.nn.init as init
import os
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
        trainset = sci.loadmat('/home/lu/code/pytorch/dataset3.mat')['train']
        self.data = trainset


    def __getitem__(self, idx):
        
        data_ori = torch.from_numpy(self.data[idx])
        data = data_ori[:153600]/20
        label = data_ori[-1]-1

        return data, label

    def __len__(self):
        return len(self.data)

trainset = TrainDataset()
train_loader = Data.DataLoader(trainset, batch_size = 1,shuffle = False)

################################################################################

class TestDataset(Data.Dataset):
    def __init__(self):
        a=sci.loadmat('/home/lu/code/pytorch/dataset3.mat')['te']
        #b=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data11.mat')['test_data1'])
        self.data = a

    def __getitem__(self, idx):
        
        data_ori = torch.from_numpy(self.data[idx])
        data = data_ori[:153600]
        label = data_ori[-1]-1
        #print(data.size())
        print(label)
        return data, label

    def __len__(self):
        return len(self.data)

testset = TestDataset()
test_loader = Data.DataLoader(testset, batch_size = 1, shuffle = False)

##################################################
class CNN(nn.Module):
    def __init__(self, hidden_size, output_size):
    #def __init__(self):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(1,8,(5,5))
        self.conv2 = nn.Conv2d(30,16,100)
        self.conv3 = nn.Conv2d(16,16,100)
        #self.conv4 = nn.Conv1d(3,1,10)

        self.fc1 = nn.Linear(2040, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, x): 
        x = init.normal(x)
        cnn_out = F.max_pool2d(F.tanh(self.conv1(x)),(10,50))
        #cnn_out = F.max_pool1d(F.relu(self.conv2(cnn_out)),2)
        #cnn_out = F.max_pool1d(F.relu(self.conv3(cnn_out)),2)
        #cnn_out = F.max_pool1d(F.sigmoid(self.conv4(cnn_out)),2)

        cnn_out = cnn_out.view(-1,2040)
        output = F.tanh(self.fc1(cnn_out))
        output = self.out(output)

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

    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs+1):
        for step1,(batch_x, batch_y) in enumerate(train_loader):

            batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
            batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))

            loss = train(batch_x.view(1,1,2560,60).transpose(2,3), batch_y, cnn, cnn_optimizer, criterion)
            
            current_loss += loss


        for step2,(test_x, test_y) in enumerate(test_loader):

            test_x = Variable(test_x.type('torch.cuda.FloatTensor'))
            test_y = test_y.type('torch.cuda.LongTensor')

            guess = test(test_x.view(1,1,2560,60).transpose(2,3), cnn)
            print(guess,test_y[0])
            confusion[guess][test_y[0]] += 1
            
            
            if step2 == 100:
                print(confusion)
                break

            
        sen = (confusion[0][0])/((confusion[0][0]+confusion[0][1]+1))
        acc = (confusion[0][0]+confusion[1][1]+confusion[2][2]+confusion[3][3])/step2
       
        all_losses.append(current_loss / step1)
        #err_rate.append(acc*100)


        print('%d epoch: acc = %.2f, sen = %.2f%%'%(epoch, acc*100, sen*100))
        current_loss = 0
        err = 0   
        confusion = torch.zeros(4,4)

cnn = CNN(500,4)

if use_cuda:
    cnn = cnn.cuda()

trainIters(cnn, 30)

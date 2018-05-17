import torch
import math
import numpy as np
import scipy.io as sci
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from PIL import Image
import torchvision.transforms as transforms

use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)
####################dataset######################################
def default_loader(path):
    return Image.open(path).convert('RGB')

def Brelu(x):
    out = min(max(x,0),1)
    return  out

class TrainDataset(Data.Dataset):
    def __init__(self, loader = default_loader):
        self.data_files = os.listdir('/home/lu/code/data/data_h')
        self.loader = loader
    def __getitem__(self, idx):
        transform1 = transforms.Compose([transforms.ToTensor()])
        num = self.data_files[idx]
        data = self.loader('/home/lu/code/data/data_h/'+str(num))   
        data1 = transform1(data)
        label = 0.1*int(num[-5])

        label = torch.ones(16,16)*label
        return data1, label

    def __len__(self):
        return 130410

trainset = TrainDataset()

train_loader = Data.DataLoader(trainset, batch_size = 128, shuffle = True)

class TestDataset(Data.Dataset):
    def __init__(self, loader = default_loader):
        self.data_files = os.listdir('/home/lu/code/data/h2')
        self.loader = loader
    def __getitem__(self, idx):
        transform1 = transforms.Compose([transforms.ToTensor()])
        num = self.data_files[idx]

        data = self.loader('/home/lu/code/data/h2/'+str(num))   
        data1 = transform1(data)
        label = 0.1*int(num[-5])

        label = torch.ones(240,320)*label
        return data1, label

    def __len__(self):
        return 18

testset = TestDataset()

test_loader = Data.DataLoader(testset, batch_size = 1, shuffle = True)

##################network###########################################
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,16,(7,7), padding = 3)
        self.maxpool = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.maxout = nn.Conv2d(16,4,(5,5), padding = 2)

        self.conv2 = nn.Conv2d(4,16,(1,1),padding = 0)
        self.conv3 = nn.Conv2d(4,16,(3,3),padding = 1)
        self.conv4 = nn.Conv2d(4,16,(5,5),padding = 2)
        self.conv5 = nn.Conv2d(4,16,(7,7),padding = 3)

        #self.avgpool = nn.AvgPool2d(3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(64,1,(3,3),stride = 1, padding = 1)
        self.brelu = nn.ReLU()

    def forward(self, x):

        cnn_out = self.conv1(x)
        #cnn_out = self.maxpool(cnn_out)
        
        
        cnn_out = self.maxout(cnn_out)
        cnn_out = self.maxpool(cnn_out)

        t1 = self.conv2(cnn_out)
        t2 = self.conv3(cnn_out)
        t3 = self.conv4(cnn_out)
        t4 = self.conv5(cnn_out)

        cnn_out = torch.cat((t1,t2,t3,t4),1)
        cnn_out = self.maxpool(cnn_out)
        
        cnn_out = self.conv6(cnn_out)
        cnn_out = self.brelu(cnn_out)   

        return cnn_out

######################train#####################################

def train(input_variable, target_variable, encoder, encoder_optimizer, criterion):

    encoder_optimizer.zero_grad()
    loss = 0
    encoder_outputs = encoder(input_variable)

    loss = criterion(encoder_outputs, target_variable)

    loss.backward()
    encoder_optimizer.step()

    return loss.data[0]

def test(input_variable, encoder):

    encoder_outputs = encoder(input_variable)
        
    return encoder_outputs

#################################################################
def trainIters(cnn, epoch, learning_rate=0.001):  

    n_epochs = epoch
    current_loss = 0

    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum = 0.9)

    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs+1):
        for step1,(batch_x, batch_y) in enumerate(train_loader):
            #print(batch_x,batch_y)
            batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
            batch_y = Variable(batch_y.type('torch.cuda.FloatTensor'))

            loss = train(batch_x, batch_y, cnn, cnn_optimizer,  criterion)
            
            current_loss += loss

        for step2,(batch_x, batch_y) in enumerate(test_loader):
            print(batch_x,batch_y)
            batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
            batch_y = Variable(batch_y.type('torch.cuda.FloatTensor'))

            print(test(batch_x,cnn))
            break

        print('Epoch:',epoch, '|train loss: %.4f' %(current_loss/step1) )
        current_loss = 0

cnn = CNN()

if use_cuda:
    cnn = cnn.cuda()
 
trainIters(cnn, 50)

torch.save(cnn, '/home/lu/code/data/DeHaze2.pkl')
torch.save(cnn.state_dict(), '/home/lu/code/data/DeHaze_params2.pkl')
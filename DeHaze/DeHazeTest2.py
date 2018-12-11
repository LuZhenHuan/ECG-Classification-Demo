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
torch.cuda.set_device(1)
####################dataset######################################
def default_loader(path):
    return Image.open(path).convert('RGB')

class TestDataset(Data.Dataset):
    def __init__(self, loader = default_loader):
        self.data_files = os.listdir('/home/lu/code/data/h')
        self.loader = loader
    def __getitem__(self, idx):
        transform1 = transforms.Compose([transforms.ToTensor()])
        num = self.data_files[idx]
        print(num)
        data = self.loader('/home/lu/code/data/h/'+str(num))   
        data1 = transform1(data)
        label = 0.1*int(num[-5])

        label = torch.ones(240,320)*label
        return data1, label

    def __len__(self):
        return 13041

testset = TestDataset()

test_loader = Data.DataLoader(testset, batch_size = 1, shuffle = True)

##################network###########################################
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,5,(7,7), padding = 3)
        self.maxpool = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(5,5,(3,3), padding = 1)

        self.conv3 = nn.Conv2d(5,5,(1,1),padding = 0)
        self.conv4 = nn.Conv2d(5,5,(3,3),padding = 1)
        self.conv5 = nn.Conv2d(5,5,(5,5),padding = 2)

        self.avgpool = nn.AvgPool2d(3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(20,1,(3,3),stride = 1, padding = 1)
        self.brelu = nn.ReLU()

    def forward(self, x):

        cnn_out = self.conv1(x)
        cnn_out = self.maxpool(cnn_out)
        
        cnn_out = self.conv2(cnn_out)
        cnn_out = self.maxpool(cnn_out)

        t1 = self.conv3(cnn_out)
        t2 = self.conv4(cnn_out)
        t3 = self.conv5(cnn_out)
        t4 = self.maxpool(cnn_out)

        cnn_out = torch.cat((t1,t2,t3,t4),1)
        cnn_out = self.avgpool(cnn_out)
        
        cnn_out = self.conv6(cnn_out)
        cnn_out = self.brelu(cnn_out)   ###How to use BReLU

        return cnn_out

encoder = torch.load('/home/lu/code/data/DeHaze2.pkl').cuda()
#decoder = torch.load('/home/lu/code/pytorch/data_dir/decoder.pkl').cuda()

def test(input_variable, encoder):

    encoder_outputs = encoder(input_variable)
        
    return encoder_outputs

for step1,(batch_x, batch_y) in enumerate(test_loader):
        print(batch_x,batch_y)
        batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
        batch_y = Variable(batch_y.type('torch.cuda.FloatTensor'))

        print(test(batch_x,encoder))
        break
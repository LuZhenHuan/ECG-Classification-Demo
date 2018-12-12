##################################################################
# prepare our data, we first to build a test dataset which is like 
# the MIT ecg data. it is <5x1x400>
#自己搭建简易rnn

import torch
import time
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.serialization import load_lua

N, T ,D= 50, 5, 400	#opt.batch_size, opt.seq_length , word_dim	

train_temp = load_lua('/home/lu/code/D1Train.t7')
trainset = train_temp.view(50,-1,2000).transpose(0,1).clone()
data_len = trainset.size()[0]

test_temp = load_lua('/home/lu/code/D1Test.t7')
testset = test_temp.view(-1,2000)
test_len = testset.size()[0]
print(data_len, test_len)

count = 0

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

def read_data():
    global count, trainset

    x = trainset[count].view(N,T,D).transpose(0,1).clone()
    x = x.type('torch.FloatTensor')

    count +=  1
    if(count == data_len):
        count = 0

    y = torch.LongTensor(50)
    y[0:25] = 0
    y[25:50] = 1

    return Variable(x), Variable(y)

##################################################################
# build a nerul network 
# a handcraft RNN, it can easily implement by pytorch
# keypoint is cloning the parameters of a layer over several timestep

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        #self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = F.relu(self.i2h(combined))
        output = F.relu(self.i2o(combined))
        #output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(N, self.hidden_size))

##################################################################
# train loop & test loop
model = RNN(400, 100, 2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(input,label):
    hidden = model.initHidden()
    optimizer.zero_grad()
    for i in range(input.size()[0]):
        output, hidden = model(input[i], hidden)
    loss = criterion(output, label)
    loss.backward()

    optimizer.step()
    return output, loss.data[0]

def test(input):
    hidden = Variable(torch.zeros(1,100))
    input = Variable(input.view(5,1,400))
    input = input.type('torch.FloatTensor')

    for i in range(input.size()[0]):
        output, hidden = model(input[i], hidden)
    top_n, top_i = output.data.topk(1)
    
    return top_i[0][0]

##################################################################
# let's train it

n_epochs = 2
print_every = 1
current_loss = 0
all_losses = []
err_rate = []
err = 0

for epoch in range(1, 50):
    input, target = read_data()
    output, loss = train(input, target)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        print(epoch, current_loss / print_every)
        all_losses.append(current_loss / print_every)
        current_loss = 0
        
        for i in range(test_len):
            guess = test(testset[i])
            if i < test_len/2 and guess == 1:
                err +=1
            if i >= test_len/2 and guess == 0:
                err += 1
        err_rate.append((1-err/test_len)*100)

        print(err)
        print((1-err/test_len)*100)
        err = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.title('loss')
plt.figure()
plt.plot(err_rate)
plt.title('err')
print(timeSince(start))

plt.show()

print(err)    







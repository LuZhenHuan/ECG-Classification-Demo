import torch
import math
import scipy.io as sci
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.serialization import load_lua

N, T ,D= 60, 5, 400	#opt.batch_size, opt.seq_length , word_dim	

#process traindata and trainlabel
train_temp = torch.from_numpy(sci.loadmat('/home/lu/code/pytorch/D1TrainCCDD.mat')['trainset']*0.0048).clone()
trainset = train_temp.view(N,-1,4000).transpose(0,1)
data_len = trainset.size()[0]

train_label = torch.cuda.LongTensor(64800)
for i in range(6):
    train_label[i*10800:(i+1)*10800] = i

train_label = train_label.view(-1,N)

y = torch.cuda.LongTensor(60)
for i in range(6):
    y[i*10:(i+1)*10] = i


#process testdata and testlabel
test_temp = torch.from_numpy(sci.loadmat('/home/lu/code/pytorch/D1TestCCDD.mat')['testset']*0.0048).clone()
testset = test_temp.view(-1,1,4000).cuda()
test_len = testset.size()[0]

test_label = torch.cuda.LongTensor(7200)
for i in range(6):
    test_label[i*1200:(i+1)*1200] = i

test_dataset = {'data':testset,'label':test_label}

print(data_len, test_len)

count = 0

def read_data():
    global count, trainset, y

    x = trainset[count]
    x = x.type('torch.cuda.FloatTensor')

    #y = train_label[count]

    count +=  1
    if(count == data_len):
        count = 0

    return Variable(x), Variable(y)

##################################################################
# build a MLP

class MLP(nn.Module):
    def __init__(self, input_size, hidden_szie, output_size):
        super(MLP, self).__init__()
        
        self.i2h = nn.Linear(input_size, hidden_szie1)
        self.h2o = nn.Linear(hidden_szie3, output_size)

    def forward(self, input):
        hidden1 = F.sigmoid(self.iTOh1(input))
        hidden2 = F.sigmoid(self.h1TOh2(hidden1))
        hidden3 = F.sigmoid(self.h2TOh3(hidden2))
        output = self.h3TOo(hidden3)

        return output

##################################################################
# train loop
model = MLP(4000, 3000, 2000, 1000, 6)
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

    input = Variable(input)
    input = input.type('torch.cuda.FloatTensor')

    output = model(input)
    top_n, top_i = output.data.topk(1)
    
    return top_i[0][0]

##################################################################
# let's train it

n_epochs = 200
print_every = data_len
current_loss = 0
all_losses = []
err_rate = []
err = 0

for epoch in range(1, data_len*n_epochs+1):
    input, target = read_data()
    output, loss = train(input, target)
    current_loss += loss

    if epoch % print_every == 0:
        print(math.floor(epoch / data_len), current_loss / print_every)
        all_losses.append(current_loss / print_every)
        current_loss = 0
        if epoch >= data_len*1:
            for i in range(test_len):
                guess = test(test_dataset['data'][i])
                #print(guess)
                if guess != test_dataset['label'][i]:
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

plt.show()

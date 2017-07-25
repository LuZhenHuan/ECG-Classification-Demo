import torch
import math
import numpy as np
import scipy.io as sci
import torch.nn as nn
import h5py
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

N, T ,D= 50, 16, 250	#opt.batch_size, opt.seq_length , word_dim	

#process traindata and trainlabel
train_temp=torch.from_numpy(np.transpose(np.array(h5py.File('/home/lu/code/pytorch/CCDD2Class.mat')['trainset'])))
trainset=train_temp*0.0048
train_len = trainset.size(0)

train_label = torch.cuda.LongTensor(96000)
for i in range(2):
    train_label[i*48000:(i+1)*48000] = i

train_dataset = Data.TensorDataset(data_tensor=trainset, target_tensor=train_label)

train_loader = Data.DataLoader(
    dataset = train_dataset,
    batch_size = N,
    shuffle = True,
)

#process valdata
val_temp=torch.from_numpy(np.transpose(np.array(h5py.File('/home/lu/code/pytorch/CCDD2Class.mat')['valset'])))
valset=val_temp*0.0048
val_len = valset.size(0)

val_label = torch.cuda.LongTensor(12000)
for i in range(2):
    val_label[i*6000:(i+1)*6000] = i

val_dataset = Data.TensorDataset(data_tensor=valset, target_tensor = val_label)

val_loader = Data.DataLoader(
    dataset = val_dataset,
    batch_size = N,
    shuffle = True,
)

#process testdata and testlabel
test_temp=torch.from_numpy(np.transpose(np.array(h5py.File('/home/lu/code/pytorch/CCDD2Class.mat')['testset'])))
testset=test_temp*0.0048
test_len = testset.size(0)

test_label = torch.cuda.LongTensor(12000)
for i in range(2):
    test_label[i*6000:(i+1)*6000] = i

test_dataset = {'data':testset,'label':test_label}

print(train_len, val_len, test_len)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_szie, output_size):
        super(RNN, self).__init__() 
        self.rnn = nn.LSTM(input_size, hidden_szie, 1, dropout = 0.5, bidirectional = True)
        self.h2o = nn.Linear(hidden_szie*2, output_size)

    def forward(self, input):
        hidden, _ = self.rnn(input)
        output = self.h2o(hidden[T-1])
        return output

##################################################################
# train loop
model = RNN(D, D, 2)
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
    #hidden = Variable(torch.zeros(1,100))
    input = Variable(input.view(T,1,D))
    input = input.type('torch.cuda.FloatTensor')

    output = model(input)
    top_n, top_i = output.data.topk(1)
    
    return top_i[0][0]


##################################################################
# let's train it

n_epochs = 20
current_loss = 0
val_loss = 0
all_losses = []
val_losses = []
err_rate = []
err = 0
confusion = torch.zeros(2,2)

for epoch in range(1, n_epochs+1):
    for step1,(batch_x, batch_y) in enumerate(train_loader):
        batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
        batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))
        output, loss = train(batch_x.view(N,T,D).transpose(0,1), batch_y)
        current_loss += loss

    for step2,(batch_x, batch_y) in enumerate(val_loader):
        batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
        batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))
        
        output = model(batch_x.view(N,T,D).transpose(0,1))
        valloss = criterion(batch_x, batch_y)
        val_loss += valloss.data[0]

    for i in range(test_len):
        guess = test(test_dataset['data'][i])
        #print(guess)
        if guess != test_dataset['label'][i]:
                err += 1
        if epoch == n_epochs:
            confusion[guess][test_dataset['label'][i]] += 1
    
    print('epoch',epoch,)

    all_losses.append(current_loss / step1)
    val_losses.append(val_loss / step2)
    current_loss = 0
    val_loss = 0
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
plt.plot(val_losses)

plt.figure()
plt.plot(err_rate)
plt.title('err')

print(confusion)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

plt.show()

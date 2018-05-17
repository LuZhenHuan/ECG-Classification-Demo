import torch
import math
import torch.nn as nn
import torchvision
import scipy.io as sci
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import random

from torch.utils.serialization import load_lua
torch.cuda.set_device(1)

N, T ,D= 50, 5, 400	#opt.batch_size, opt.seq_length , word_dim	

train_temp = torch.from_numpy(sci.loadmat('/home/lu/code/MITcvNew/D5Train.mat')['trainset']).clone()
train_temp = train_temp/4
trainset = train_temp.view(-1,2000)
data_len = trainset.size()[0]

train_label = torch.cuda.LongTensor(trainset.size()[0])
for i in range(2):
    train_label[i*13000:(i+1)*13000] = i

train_dataset = Data.TensorDataset(data_tensor=trainset, target_tensor=train_label)

train_loader = Data.DataLoader(
    dataset = train_dataset,
    batch_size = N,
    shuffle = True,
    drop_last = True,
)

test_temp = torch.from_numpy(sci.loadmat('/home/lu/code/MITcvNew/D5Test.mat')['testset']).clone()
test_temp = test_temp/4
testset = test_temp.view(-1,1,2000).cuda()
test_len = testset.size()[0]


test_label = torch.cuda.LongTensor(test_len)
for i in range(2):
    test_label[i*1625:(i+1)*1625] = i

test_dataset = {'data':testset,'label':test_label}

print(data_len, test_len)

##################################################################
# build a nerul network with nn.RNN

class RNN1(nn.Module):
    def __init__(self, input_size, hidden_szie1, hidden_szie2, output_size):
        super(RNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_szie1, 4, dropout = 0.4)
        self.r2h = nn.Linear(hidden_szie1, hidden_szie2)
        self.h2o = nn.Linear(hidden_szie2, output_size)

    def forward(self, input):
        hidden, _ = self.rnn(input)
        fc1 = F.relu(self.r2h(hidden[T-1]))
        output = self.h2o(fc1)

        return output

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout = 0.3, max_length = T):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.dropout = dropout
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
        output = self.attn_combine(attn_applied)
        output = F.relu(output)
        output = self.out(output)
        #output = F.softmax(output)
        #print(output)
        return output

##################################################################
# train loop
model = RNN(D, 400,  200, 2)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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

n_epochs = 50
print_every = data_len
current_loss = 0
all_losses = []
err_rate = []
err = 0
confusion = torch.zeros(2,2)

for epoch in range(1, n_epochs+1):
    for step1,(batch_x, batch_y) in enumerate(train_loader):
        batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
        batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))
        output, loss = train(batch_x.view(N,T,D).transpose(0,1), batch_y)
        current_loss += loss

    for i in range(test_len):
        guess = test(test_dataset['data'][i])
        confusion[guess][test_dataset['label'][i]] += 1

    if epoch == 1000:
        k = random.randint(1,1000)
        sample = test_dataset['data'][k]
        guess = test(test_dataset['data'][i])
        feature = hidden.data.view(-1).type('torch.FloatTensor')
        sample = sample.view(-1).type('torch.FloatTensor')
        
        testsample = open('testsample.csv', 'w+')
        for item in range(len(sample)):
            testsample.write(str(sample[item]) + '\n')

        testsample.close()

        LSTMfeature = open('LSTMfeature.csv', 'w+')
        for item in range(len(feature)):
            LSTMfeature.write(str(feature[item]) + '\n')

        LSTMfeature.close()

        print(sample, feature)

    
    sen = (confusion[0][0])/((confusion[0][0]+confusion[0][1]))
    acc = (confusion[0][0]+confusion[1][1])/test_len
    
    all_losses.append(current_loss / step1)
    err_rate.append(acc*100)
    
    current_loss = 0
    print('%d epoch: acc = %.2f, sen = %.2f%%'%(epoch, acc*100, sen*100))
    err = 0   
    confusion = torch.zeros(2,2)

#print(confusion)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


plt.figure()
plt.plot(all_losses)
plt.title('loss')
plt.figure()
plt.plot(err_rate)
plt.title('err')

#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(confusion.numpy())
#fig.colorbar(cax)

plt.show()


RNN_loss = open('temp.csv', 'w+')
for item in range(len(all_losses)):
    RNN_loss.write(str(all_losses[item]) + '\n')

RNN_loss.close()
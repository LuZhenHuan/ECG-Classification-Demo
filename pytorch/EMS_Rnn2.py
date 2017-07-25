import torch
import math
import torch.nn as nn
import torchvision
import scipy.io as sci
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

from torch.utils.serialization import load_lua

N, T ,D= 50, 5, 400	#opt.batch_size, opt.seq_length , word_dim	

train_temp = torch.from_numpy(sci.loadmat('/home/lu/code/MITcvNew/D9Train.mat')['trainset']).clone()
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

test_temp = torch.from_numpy(sci.loadmat('/home/lu/code/MITcvNew/D9Test.mat')['testset']).clone()
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

class RNN(nn.Module):
    def __init__(self, input_size, hidden_szie1, hidden_szie2, output_size):
        super(RNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_szie1, 4, dropout = 0.1)
        self.r2h = nn.Linear(hidden_szie1, hidden_szie2)
        self.h2o = nn.Linear(hidden_szie2, output_size)

    def forward(self, input):
        hidden, _ = self.rnn(input)
        fc1 = F.relu(self.r2h(hidden[T-1]))
        output = self.h2o(fc1)
        return output

##################################################################
# train loop
model = RNN(D, 100,  50, 2)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
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
        #print(guess)
        if guess != test_dataset['label'][i]:
                err += 1
        if epoch == n_epochs:
            confusion[guess][test_dataset['label'][i]] += 1

    all_losses.append(current_loss / step1)
    current_loss = 0
    err_rate.append((1-err/test_len)*100)
    print('%d epoch: err numble = %d, err rate = %.2f%%'%(epoch, err, ((1-err/test_len)*100)))
    err = 0   

print(confusion)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


plt.figure()
plt.plot(all_losses)
plt.title('loss')
plt.figure()
plt.plot(err_rate)
plt.title('err')

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

plt.show()

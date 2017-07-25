import torch
import math
import scipy.io as sci
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

N, T ,D= 60, 8, 500	#opt.batch_size, opt.seq_length , word_dim	

trainset = torch.from_numpy(sci.loadmat('/home/lu/code/pytorch/D1TrainCCDD.mat')['trainset']*0.0048).clone()
train_len = trainset.size()[0]

train_label = torch.cuda.LongTensor(train_len)
for i in range(6):
    train_label[i*10800:(i+1)*10800] = i

train_dataset = Data.TensorDataset(data_tensor=trainset, target_tensor=train_label)

train_loader = Data.DataLoader(
    dataset = train_dataset,
    batch_size = N,
    shuffle = True,
)

#process testdata and testlabel
testset = torch.from_numpy(sci.loadmat('/home/lu/code/pytorch/D1TestCCDD.mat')['testset']*0.0048).clone()
test_len = testset.size()[0]

test_label = torch.cuda.LongTensor(7200)
for i in range(6):
    test_label[i*1200:(i+1)*1200] = i

test_dataset = {'data':testset,'label':test_label}

##################################################################
# build a nerul network with nn.RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_szie, output_size):
        super(RNN, self).__init__() 
        self.rnn = nn.LSTM(input_size, hidden_szie, 4)
        self.h2o = nn.Linear(hidden_szie, output_size)

    def forward(self, input):
        hidden, _ = self.rnn(input)
        output = self.h2o(hidden[T-1])
        return output

##################################################################
# train loop
model = RNN(D, 250, 6)
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

n_epochs = 50
current_loss = 0
all_losses = []
err_rate = []
confusion = torch.zeros(6, 6)
err = 0

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

    all_losses.append(current_loss / step1+1)

    err_rate.append((1-err/test_len)*100)

    print('%d epoch: err numble = %d, err rate = %.2f%%'%(epoch, err, ((1-err/test_len)*100)))
    
    err = 0
    current_loss

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

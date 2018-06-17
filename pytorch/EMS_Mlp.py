import torch
import math
import torch.nn as nn
import scipy.io as sci
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data

from torch.utils.serialization import load_lua
torch.cuda.set_device(1)

N, T ,D= 50, 1, 2000	#opt.batch_size, opt.seq_length , dimention	

##################################################################
#整理训练集和测试集
train_temp = torch.from_numpy(sci.loadmat('/home/lu/code/MITcvNew/D9Train.mat')['trainset']).clone()
train_temp = train_temp/4
trainset = train_temp.view(-1,D)
data_len = trainset.size()[0]

train_label = torch.cuda.LongTensor(trainset.size()[0])
for i in range(2):
    train_label[i*13000:(i+1)*13000] = i

train_dataset = Data.TensorDataset(data_tensor=trainset, target_tensor=train_label)

train_loader = Data.DataLoader(
    dataset = train_dataset,
    batch_size = N,
    shuffle = True,
)

test_temp = torch.from_numpy(sci.loadmat('/home/lu/code/MITcvNew/D9Test.mat')['testset']).clone()
test_temp = test_temp/4
testset = test_temp.view(-1,1,D).cuda()
test_len = testset.size()[0]


test_label = torch.cuda.LongTensor(test_len)
for i in range(2):
    test_label[i*1625:(i+1)*1625] = i

test_dataset = {'data':testset,'label':test_label}

print(data_len, test_len)


##################################################################
# build a MLP

class MLP(nn.Module):
    def __init__(self, input_size, hidden_szie, output_size):
        super(MLP, self).__init__()
        
        self.i2h = nn.Linear(input_size, hidden_szie)
        self.h2o = nn.Linear(hidden_szie, output_size)

    def forward(self, input):
        hidden = F.sigmoid(self.i2h(input))
        output = self.h2o(hidden)
        return output

##################################################################
# train loop
model = MLP(D, 1000, 2)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(input, label):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, label)
    print(loss)
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

n_epochs = 50
print_every = data_len
current_loss = 0
all_losses = []
val_losses = []
err_rate = []
err = 0
confusion = torch.zeros(2,2)

for epoch in range(1, n_epochs+1):
    for step1,(batch_x, batch_y) in enumerate(train_loader):
        batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
        batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))
        output, loss = train(batch_x, batch_y)
        current_loss += loss

    for i in range(test_len):
        guess = test(test_dataset['data'][i])
        #print(guess)
        confusion[guess][test_dataset['label'][i]] += 1
    
    sen = (confusion[0][0])/((confusion[0][0]+confusion[0][1]))
    acc = (confusion[0][0]+confusion[1][1])/test_len

    all_losses.append(current_loss / step1)
    err_rate.append(acc*100)
    
    current_loss = 0
    print('%d epoch: acc = %.2f, sen = %.2f%%'%(epoch, acc*100, sen*100))
    err = 0   
    confusion = torch.zeros(2,2)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


plt.figure()
plt.plot(all_losses)
plt.title('loss')
plt.figure()
plt.plot(err_rate)
plt.title('err')

plt.show()

RNN_loss = open('temp.csv', 'w+')
for item in range(len(all_losses)):
    RNN_loss.write(str(all_losses[item]) + '\n')

RNN_loss.close()
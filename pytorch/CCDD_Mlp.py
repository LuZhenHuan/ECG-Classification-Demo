import torch
import math
import scipy.io as sci
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

N, T ,D= 60, 5, 400	#opt.batch_size, opt.seq_length , word_dim	

#process traindata and trainlabel
train_temp = torch.from_numpy(sci.loadmat('/home/lu/code/MITcvNew/D1Train.mat')['trainset']).clone()

train_label = torch.cuda.LongTensor(64800)
for i in range(6):
    train_label[i*10800:(i+1)*10800] = i

ecg_dataset = Data.TensorDataset(data_tensor=train_temp, target_tensor=train_label)

train_loader = Data.DataLoader(
    dataset = ecg_dataset,
    batch_size = 64,
    shuffle = True,
)

#process testdata and testlabel
test_temp = torch.from_numpy(sci.loadmat('/home/lu/code/pytorch/D1TestCCDD.mat')['testset']*0.0048).clone()
testset = test_temp.view(-1,1,4000).cuda()
test_len = testset.size()[0]

test_label = torch.cuda.LongTensor(7200)
for i in range(6):
    test_label[i*1200:(i+1)*1200] = i

test_dataset = {'data':testset,'label':test_label}

##################################################################
# build a MLP

class MLP(nn.Module):
    def __init__(self, input_size, hidden_szie, output_size):
        super(MLP, self).__init__()
        
        self.i2h = nn.Linear(input_size, hidden_szie)
        self.h2o = nn.Linear(hiddeimport scipy.io as sci
n_szie, output_size)

    def forward(self, input):
        hidden = F.sigmoid(self.i2h(input))
        output = self.h2o(hidden)

        return output

##################################################################
# train loop
model = MLP(4000, 2000, 6)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
import scipy.io as sci

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

n_epochs = 10
current_loss = 0
all_losses = []
err_rate = []
err = 0
confusion = torch.zeros(6, 6)

for epoch in range(1, n_epochs):
    for step,(batch_x, batch_y) in enumerate(train_loader):
        batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
        batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))
        output, loss = train(batch_x, batch_y)
        current_loss += loss

    print(epoch)

    all_losses.append(current_loss / 1080)
    current_loss = 0

for i in range(test_len):
    guess = test(test_dataset['data'][i])
    #print(guess)
    if guess != test_dataset['label'][i]:
            err += 1
        
    confusion[guess][test_dataset['label'][i]] += 1

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

print(confusion)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

plt.show()

import torch
import math
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.serialization import load_lua

N, T ,D= 50, 5, 400	#opt.batch_size, opt.seq_length , word_dim	

train_temp = load_lua('D1Train.t7')
trainset = train_temp.view(50,-1,2000).transpose(0,1).clone().cuda()
data_len = trainset.size()[0]

test_temp = load_lua('D1Test.t7')
testset = train_temp.view(-1,2000).cuda()
test_len = testset.size()[0]

count = 0

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
# build a nerul network with nn.RNN

class lstm(nn.Module):
    def __init__(self, input_size, hidden_szie, output_size):
        super(lstm, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_szie, 2)
        self.h2o = nn.Linear(hidden_szie, output_size)

    def forward(self, input):
        hidden, _ = self.rnn(input)
        output = self.h2o(hidden[4])
        return output

##################################################################
# train loop
model = lstm(400, 200, 2)

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
    hidden = Variable(torch.zeros(1,100))
    input = Variable(input.view(5,1,400))
    input = input.type('torch.FloatTensor')

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

for epoch in range(1, data_len*n_epochs+1):
    input, target = read_data()
    output, loss = train(input, target)
    current_loss += loss

    if epoch % print_every == 0:
        print(math.floor(epoch / data_len), current_loss / print_every)
        all_losses.append(current_loss / print_every)
        current_loss = 0

        for i in range(test_len):
            guess = test(testset[i])
            if i < test_len/2 and guess == 1:
                    err +=1
            if i >= test_len/2 and guess == 0:
                    err += 1
                    
        err_rate.append((1-err/test_len)*100)
        if epoch / data_len >= 40:
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


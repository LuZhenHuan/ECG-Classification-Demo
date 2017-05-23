##################################################################
# prepare our data, we first to build a test dataset which is like 
# the MIT ecg data. it is <5x1x400>

import torch
import torch.nn.functional as F
from torch.autograd import Variable

test_input = Variable(torch.ones(5,2,400))
test_label = Variable(torch.LongTensor(50))

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
        return Variable(torch.zeros(2, self.hidden_size))

##################################################################
# train loop

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(input,label):
    hidden = model.initHidden()
    optimizer.zero_grad()

    for i in range(input.size()[0]):
        output, hidden = model(input[i], hidden)

    print(input,input[1],label)    
    loss = criterion(output, label)
    loss.backward()
    print(loss)
    optimizer.step()

##################################################################
#let's train it

model = RNN(400, 100, 2)

n_epochs = 5200 
print_every = 5000
plot_every = 1000
current_loss = []

for epoch in range(1, 5200 + 1):
    target, input = read_data()
    output, loss = train(input, target)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))













import torch
import math
import scipy.io as sci
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

torch.cuda.set_device(1)
use_cuda = torch.cuda.is_available()

N, T ,D= 50, 8, 500	#opt.batch_size, opt.seq_length , word_dim	

#########################################################
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

print('train_len = %d, test_len = %d'%(train_len, test_len))

##################################################
class CNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.001, max_length=L):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.dropout = nn.Dropout(self.dropout_p)

        self.conv1 = nn.Conv1d(L,L,100)
        self.conv2 = nn.Conv1d(L,L,100)
        self.conv3 = nn.Conv1d(L,1,100)
        #self.conv4 = nn.Conv1d(3,1,10)

        self.fc1 = nn.Linear(851, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_outputs):

        cnn_out = F.sigmoid(self.conv1(encoder_outputs))
        cnn_out = F.sigmoid(self.conv2(cnn_out))
        cnn_out = F.max_pool1d(F.sigmoid(self.conv3(cnn_out)),2)
        #cnn_out = F.max_pool1d(F.sigmoid(self.conv4(cnn_out)),2)
        #print(cnn_out)
        cnn_out = cnn_out.view(-1,851)
        output = F.relu(self.fc1(cnn_out))
        output = self.out(output)
        
        return output

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout = 0.3, max_length = T):
        super(Attention, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.output_size = output_size
        self.dropout = dropout

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
        
        return output, hidden, attn_weights


def train(input_variable, target_variable, cnn, rnn, cnn_optimizer, rnn_optimizer, criterion, max_length=T):

    cnn_optimizer.zero_grad()
    rnn_optimizer.zero_grad()

    loss = 0
    cnn_output = cnn(input_variable)
    cnn_output = cnn_output.view(N, T, D).transpose(0,1)
    rnn_output, hidden, attn_weight = rnn(cnn_output)
    loss = criterion(rnn_output, target_variable)
    loss.backward()

    cnn_optimizer.step()
    rnn_optimizer.step()

    return loss.data[0]

def test(input_variable, cnn, rnn):
    
    cnn_output = cnn(input_variable)
    cnn_output = cnn_output.view(1, T, D).transpose(0,1)
    rnn_output, hidden, attn_weight = rnn(cnn_output)
    top_n, top_i = rnn_output.data.topk(1)

    return top_i[0][0]

def trainIters(cnn, epoch, learning_rate=0.001):

    n_epochs = epoch
    current_loss = 0
    all_losses = []
    err_rate = []
    confusion = torch.zeros(6, 6)
    err = 0

    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs+1):
        for step1,(batch_x, batch_y) in enumerate(train_loader):

            batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
            batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))
            loss = train(batch_x.view(N,L,-1), batch_y, cnn, rnn, cnn_optimizer, rnn_optimizer, criterion)
            current_loss += loss

        for step2,(test_x, test_y) in enumerate(test_loader):

            test_x = Variable(test_x.type('torch.cuda.FloatTensor'))
            test_y = test_y.type('torch.cuda.LongTensor')
            guess = test(test_x.view(1,L,-1), cnn, rnn)
            #print('g',guess,'t',test_y[0])
            if guess != test_y[0]:
                    err += 1
        
            if epoch == n_epochs:
                confusion[guess][test_y[0]] += 1   

        print(current_loss/(step1+1))
        all_losses.append(current_loss/(step1+1))
        err_rate.append((1-err/step2)*100)
        print(err)
        print('%d epoch:, err number = %d, err rate = %.2f%%'%(epoch, err, ((1-err/step2)*100)))
    
        current_loss = 0
        err = 0

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


cnn = CNN(400, 200)
rnn = Attention(10, hidden_size, O)

if use_cuda:
    cnn = cnn.cuda()
    rnn =rnn.cuda()

trainIters(cnn, 50)

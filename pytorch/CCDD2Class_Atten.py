import torch
import math
import numpy as np
import scipy.io as sci
import torch.nn as nn
import h5py
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

torch.cuda.set_device(1)
use_cuda = torch.cuda.is_available()

N, T ,D= 100, 10, 200	#opt.batch_size, opt.seq_length , word_dim	


###############################################################################
class TrainDataset(Data.Dataset):
    def __init__(self):
        self.data = torch.from_numpy(np.transpose(np.array(h5py.File('/home/lu/code/pytorch/data_dir/data1_3.mat')['traindata'])))   

        #self.data_files = os.listdir('/home/lu/code/pytorch/data_dir/Train')
        #self.train_label = np.loadtxt(open('/home/lu/code/pytorch/data_dir/train_label.csv','rb'))   

    def __getitem__(self, idx):
        
        data_ori = self.data[idx]
        #num = int(self.data_files[idx].split('.')[0])
        #label = self.train_label[num-1]
        data = data_ori[2000:4000]
        label = data_ori[24001]
        return data, label

    def __len__(self):
        return len(self.data)

trainset = TrainDataset()

train_loader = Data.DataLoader(trainset, batch_size = N,shuffle = True)

################################################################################
class TestDataset(Data.Dataset):
    def __init__(self):
        self.data = torch.from_numpy(np.transpose(np.array(h5py.File('/home/lu/code/pytorch/data_dir/data1_3.mat')['testdata'])))   
        #self.test_label = np.loadtxt(open('/home/lu/code/pytorch/data_dir/test_label.csv','rb'))   

    def __getitem__(self, idx):
        
        data_ori = self.data[idx]
        #num = int(self.data_files[idx].split('.')[0])
        #label = self.test_label[num-1]
        data = data_ori[2000:4000]
        label = data_ori[24001]
        
        return data, label

    def __len__(self):
        return len(self.data)

testset = TestDataset()

test_loader = Data.DataLoader(testset, batch_size = 1, shuffle = True)

##################################################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, bidirectional = True)

    def forward(self, input):
        
        output, hidden = self.gru(input)

        output = output.transpose(0,1)
        hidden = torch.cat((hidden[0],hidden[1]),1)

        return output, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.001, max_length=T):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, hidden, encoder_outputs):

        attn_weights = self.attn(hidden)
        attn_weights = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attn_weights = F.softmax(self.attn(attn_weights.squeeze(1)))

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attn_applied = attn_applied.squeeze(1)
        #output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(attn_applied)
        output = F.relu(output)
        output = self.out(output)
        #output = F.log_softmax(self.out(output[0]))
        
        return output, hidden, attn_weights


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=T):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_variable)
    
    decoder_hidden = encoder_hidden
    
    decoder_output, decoder_hidden, attn_weights = decoder(decoder_hidden, encoder_outputs)

    loss += criterion(decoder_output, target_variable)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]

def test(input_variable, encoder, decoder):

    encoder_outputs, encoder_hidden = encoder(input_variable)
    
    decoder_hidden = encoder_hidden

    decoder_output, decoder_hidden, attn_weights = decoder(decoder_hidden, encoder_outputs)
    
    top_n, top_i = decoder_output.data.topk(1)
    
    return top_i[0][0], attn_weights

def trainIters(encoder, decoder, learning_rate=0.001):

    n_epochs = 1
    current_loss = 0
    all_losses = []
    err_rate = []
    confusion = torch.zeros(6, 6)
    err = 0

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs+1):
        for step1,(batch_x, batch_y) in enumerate(train_loader):
            batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
            batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))

            loss = train(batch_x.view(N,T,D).transpose(0,1), batch_y, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
            
            current_loss += loss
        
        for step2,(test_x, test_y) in enumerate(test_loader):
            
            test_x = Variable(test_x.type('torch.cuda.FloatTensor'))
            test_y = test_y.type('torch.cuda.LongTensor')
            guess, attn_weight = test(test_x.view(1,T,-1).transpose(0,1), encoder, decoder)
            #guess = test(test_dataset['data'][i], encoder, decoder)
            #print(guess, test_y)       
            if guess != test_y[0]:
                    err += 1
        
            if epoch == n_epochs:
                confusion[guess][test_y[0]] += 1   
                
                top_n, top_i = attn_weight.data.topk(1)
                num = top_i[0][0]
                print(num)
                a = test_x.data[0][num*D:(num+1)*D]
                print(type(a))
                plt.figure()
                plt.plot(a)
                break

        print(current_loss/(step1+1))
        all_losses.append(current_loss/(step1+1))
        err_rate.append((1-err/step2)*100)

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



hidden_size = 100
encoder = EncoderRNN(D, hidden_size)
decoder = AttnDecoderRNN(hidden_size, 2, 1)

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

trainIters(encoder, decoder)

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
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=T):
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
        
        attn_weights = F.softmax(self.attn(hidden))
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attn_applied = attn_applied.squeeze(1)
        #output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(attn_applied)
        output = F.sigmoid(output)
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

    input_variable = Variable(input_variable.view(T,1,D))
    input_variable = input_variable.type('torch.cuda.FloatTensor')

    encoder_outputs, encoder_hidden = encoder(input_variable)
    
    
    decoder_hidden = encoder_hidden
    
    decoder_output, decoder_hidden, attn_weights = decoder(decoder_hidden, encoder_outputs)
    
    top_n, top_i = decoder_output.data.topk(1)
    
    return top_i[0][0]

def trainIters(encoder, decoder, learning_rate=0.001):

    n_epochs = 50
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
        
        for i in range(test_len):
            guess = test(test_dataset['data'][i], encoder, decoder)
            #print(guess)
            if guess != test_dataset['label'][i]:
                    err += 1
        
            if epoch == n_epochs:
                confusion[guess][test_dataset['label'][i]] += 1   

        print(current_loss/(step1+1))
        all_losses.append(current_loss/(step1+1))
        err_rate.append((1-err/test_len)*100)

        print('%d epoch:, err number = %d, err rate = %.2f%%'%(epoch, err, ((1-err/test_len)*100)))
    
        current_loss = 0
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



hidden_size = 300
encoder = EncoderRNN(D, hidden_size)
decoder = AttnDecoderRNN(hidden_size, 6, 1)

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

trainIters(encoder, decoder)

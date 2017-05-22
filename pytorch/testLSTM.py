import torch 
import torch.nn as nn
from torch.autograd import Variable

time_steps = 5
batch_size = 1
in_size = 400
classes_no = 2

model = nn.LSTM(in_size , classes_no, 2)
input_seq = Variable(torch.randn(time_steps, batch_size, in_size))
out_seq, _ = model(input_seq)
last_out = out_seq[-1]

loss = nn.CrossEntropyLoss()
target = Variable(torch.LongTensor(batch_size).random_(0, classes_no-1))
err = loss(last_out, target)

print(out_seq, last_out, target)


print(err)
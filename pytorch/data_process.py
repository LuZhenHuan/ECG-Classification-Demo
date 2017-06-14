
# read .t7 file and plot
'''from torch.utils.serialization import load_lua
import matplotlib.pyplot as plt
import torch

N, T ,D= 50, 5, 400	#opt.batch_size, opt.seq_length , word_dim	

x = load_lua('D1Train.t7')
x = x.view(50,-1,2000).transpose(0,1).clone()

y = x[1].view(50,5,400).transpose(0,1).clone()

a = y.numpy()

plt.figure()
plt.plot(a[0][0])
plt.figure()
plt.plot(a[1][0])

plt.show()'''

#read .mat file and plot
import torch
import scipy.io as sci
import matplotlib.pyplot as plt

a = sci.loadmat('/home/lu/code/D1TestCCDD.mat')
b = sci.loadmat('D1TrainCCDD.mat')

plt.figure()
plt.plot(a['testset'][1300])
plt.figure()
plt.plot(b['trainset'][1300])


plt.show()
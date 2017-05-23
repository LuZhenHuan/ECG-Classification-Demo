
from torch.utils.serialization import load_lua
import matplotlib.pyplot as plt
import torch

N, T ,D= 50, 5, 400	#opt.batch_size, opt.seq_length , word_dim	

x = load_lua('D1Train.t7')
x = x.view(50,-1,2000).transpose(0,1)
#x = x.transpose(0,1)


a = x.numpy()

plt.figure()
plt.plot(a[9][24])
plt.figure()
plt.plot(a[9][25])

plt.show()
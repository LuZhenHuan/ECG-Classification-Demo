## importing the required packages
import torch
import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
discriminant_analysis, random_projection)
from torch.utils.serialization import load_lua

## Loading and curating the data

#train_temp = sci.loadmat('/home/lu/code/MITcv/D1Train.mat')['trainset']*0.25

train_temp = load_lua('/home/lu/code/MITcv/D7Train.t7')
trainset = train_temp.view(-1,2000).clone()
data_len = trainset.size()[0]

print(data_len)

train_label = torch.cuda.LongTensor(data_len)
for i in range(2):
    train_label[i*13000:(i+1)*13000] = i


X = trainset[12000:14000].numpy()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)
print(X_tsne[0])
print(X_tsne.shape[0])
print(train_label[13000])

plt.figure()

for i in range(X_tsne.shape[0]):
    plt.plot(X_tsne[i,0],X_tsne[i,1])
    plt.text(X_tsne[i,0],X_tsne[i,1],str(train_label[12000+i]),
    color=plt.cm.Set1(train_label[12000+i]/10.),fontdict={'weight': 'bold', 'size': 9})

plt.show()


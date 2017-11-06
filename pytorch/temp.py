import torch
import h5py
import numpy as np
import os
import torch.utils.data as Data
import scipy.io as sci

class TrainDataset(Data.Dataset):
    def __init__(self):
        self.data = torch.from_numpy(np.transpose(np.array(h5py.File('/home/lu/code/pytorch/data_dir/data1_3.mat')['traindata'])))   

        #self.data_files = os.listdir('/home/lu/code/pytorch/data_dir/Train')
        #self.train_label = np.loadtxt(open('/home/lu/code/pytorch/data_dir/train_label.csv','rb'))   

    def __getitem__(self, idx):
        
        data_ori = self.data[idx]
        #num = int(self.data_files[idx].split('.')[0])
        #label = self.train_label[num-1]
        data = data_ori[0:24000]
        label = data_ori[24001]
        return data, label

    def __len__(self):
        return len(self.data)

trainset = TrainDataset()

train_loader = Data.DataLoader(trainset, batch_size = 100)

for x,(y,z) in enumerate(train_loader):
    
    y = y.numpy()
    print(y,z)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.figure()
    plt.plot(y[18])

    plt.show()

    stop
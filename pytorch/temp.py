import torch
import h5py
import numpy as np
import os
import torch.utils.data as Data
import scipy.io as sci

class MyDataset(Data.Dataset):
    def __init__(self):
        self.data_files = os.listdir('/home/lu/code/pytorch/data_dir/trainset')
        #sorted(self.data_files)

    def __getitem__(self, idx):
        i=0
        a = np.loadtxt(open('/home/lu/code/pytorch/data_dir/trainset/'+self.data_files[idx],'rb'),delimiter=",")

        
        return a

    def __len__(self):
        return len(self.data_files)


dset = MyDataset()

loader = Data.DataLoader(dset, batch_size = 5,shuffle = True)

for x,y in enumerate(loader):

    print(x,y)

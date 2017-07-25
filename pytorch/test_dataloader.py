import torch
import scipy.io as sci
import torch.utils.data as Data


train_temp = torch.from_numpy(sci.loadmat('/home/lu/code/pytorch/D1TrainCCDD.mat')['trainset']*0.0048).clone()

train_label = torch.cuda.LongTensor(64800)
for i in range(6):
    train_label[i*10800:(i+1)*10800] = i

ecg_dataset = Data.TensorDataset(data_tensor=train_temp, target_tensor=train_label)

loader = Data.DataLoader(
    dataset = ecg_dataset,
    batch_size = 60,
    shuffle = False,
)
c=0
for a,(b,d) in enumerate(loader):
    print(a)
    print(b)
    break

trainset = torch.load('RnnTrain1D.t7'):view(-1,1)

tempTra =trainset[{{1,40000000},{}}]:view(-1)
val = trainset[{{40000001,44200000},{}}]:view(-1)

torch.save('RnnTrain1Dcut.t7',tempTra)
torch.save('RnnVal1Dcut.t7',val)
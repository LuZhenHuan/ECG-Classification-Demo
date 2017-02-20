require 'torch'
count = 1
flag = 1
N, T = 1, 100
dtype = 'torch.FloatTensor'

trainset = torch.load('RnnTrain.t7')
--load and process data
function data_process()

  xTemp = trainset[count]:view(4,-1,T)

	if count <= 50250 then
		y = torch.Tensor{0}:view(1,1,1)
	else
		y = torch.Tensor{1}:view(1,1,1)
	end
  	
    count = count + 1
end

data_process()
print(count)
print(xTemp[1])

--load a batch
function next_batch()
	
	x = xTemp[flag]:view(1,-1,100)
	flag = flag + 1
	if flag == 4 then
		flag = 1
	end
	x, y = x:type(dtype), y:type(dtype)
end
next_batch()
print(flag)
print(x,y)
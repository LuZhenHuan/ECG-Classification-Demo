require 'nn'
require 'torch'
require 'cunn'
require 'cutorch'

model = nn.Sequential()
model:add(nn.Linear(400,100))
model:add(nn.Sigmoid())
model:add(nn.Linear(100,2))

crit = nn.CrossEntropyCriterion()

trainTemp = torch.load('RnnTrain1D.t7')
data = trainTemp:view(-1,400)
train_len = data:size(1)

label = torch.Tensor(train_len,1):fill(1)
label[{{train_len/2+1,train_len},1}]=2

trainset = {data = data,label = label}

setmetatable(trainset,{   
    __index = function(t,i)  
        return {t.data[i], t.label[i]}
    end
})
--trainset.data = trainset.data:double()

function trainset:size()
    return self.data:size(1)
end

local function next_batch(dataset)

	data_len = dataset:size(1)

	x = dataset[count]:view(N,T,D)

	count = count + 1
	
	if count == data_len+1 then
		count = 1
    end

	y = torch.Tensor(N*T):fill(1)
	y[{{(N*T)/2+1,N*T}}] = 2
	
	x, y = x:type(dtype), y:type(dtype)

	return x,y
end

local function f(w)

	assert(w == params)
	grad_params:zero()
	--data_process()
	x,y = next_batch(trainset)

	local scores = model:forward(x)
	scores_view = scores:view(N*T ,-1)
	local loss = crit:forward(scores_view, y)
	local grad_scores = crit:backward(scores_view, y)
	model:backward(x, grad_scores)
	--model:resetStates()

	grad_params:clamp(-5, 5)

  return loss, grad_params
end






torch.save('EcgMLP.t7',model)

--model test
testTemp = torch.load('RnnTestCut.t7')
testset = testTemp:view(-1,400)
m = testset:size(1)

testset:cuda()

for i=1,m do
	a = model:forward(testset[i])
	
    if i <=m/2 and a[1]<a[2] then
		err = err + 1
		table.insert(err_sample_tf, x)
		
	elseif i > m/2 and a[1]>a[2] then
		err = err + 1
		table.insert(err_sample_ft, x)
		
		end
end
	
print (err/m*100 ..'%')	

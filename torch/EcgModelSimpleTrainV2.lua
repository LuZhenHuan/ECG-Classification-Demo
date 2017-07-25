--修正minibatch和BPTT

require 'torch'
require 'nn'
require 'gnuplot'
require 'optim'
require 'VanillaRNN'
require 'LSTM'
require 'EcgModelSimple'
require 'cutorch'
require 'cunn'

local cmd = torch.CmdLine()

cmd:option('-gpu', 1)
cmd:option('data', 'D1')

local opt = cmd:parse(arg)

cutorch.setDevice(opt.gpu)

dtype = 'torch.CudaTensor'
--build a simple rnn 100 -> 100 -> 1
model = nn.EcgModelSimple(400,100,2):type(dtype)
crit = nn.CrossEntropyCriterion():type(dtype)

--Set up some variables we will use below

N, T ,D= 50, 5, 400	--opt.batch_size, opt.seq_length , word_dim	
count = 1
max_epochs = 20
lr_decay_every = 5
lr_decay_factor = 0.5

grad_new = torch.Tensor(N*T,2):fill(0):type('torch.CudaTensor')
train_loss_history = {}
val_loss_history = {}
err_rate = {}
err_sample_tf = {}
err_sample_ft = {}

params, grad_params = model:getParameters()

--data process
trainTemp = torch.load('/home/lu/code/MITcv/D5Train.t7')/4
trainset = trainTemp:view(N,-1,T*D):transpose(1,2):clone()
train_len = trainset:size(1)

testTemp = torch.load('/home/lu/code/MITcv/D5Test.t7')/4
testset = testTemp:view(-1,T*D)
m = testset:size(1)

--trainset:add(-trainset:mean())
--testset:add(-testset:mean())

--valset = torch.load('RnnVal.t7'):view(-1,T*D)/4

--load a batch
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

--test
local function err_test()
	err = 0

	for i = 1, m do

		x = testset[i]:view(1,T,D):type(dtype)

		a = model:forward(x):view(-1)
		if i <= m/2 and a[T*2-1]<a[2*T] then
			err = err + 1
			table.insert(err_sample_tf, x)
			--print(i)
		elseif i > m/2 and a[T*2-1]>a[2*T] then
			err = err + 1
			table.insert(err_sample_ft, x)
			--print(i)
		end
	end
	err_r = err/m
	return err_r
end

--loss function 
local function f(w)

	assert(w == params)
	grad_params:zero()
	--data_process()
	x,y = next_batch(trainset)

	local scores = model:forward(x)
	scores_view = scores:view(N*T ,-1)

	local loss = crit:forward(scores_view, y)
	local grad_scores = crit:backward(scores_view, y)
	
	a = grad_scores:clone()
	for i = 1, N*T do 
		if i%T == 0 then
			grad_new[{{i}}]=a[{{i}}]
		end
	end

	model:backward(x, grad_new)
	--model:resetStates()

	grad_params:clamp(-5, 5)

  return loss, grad_params
end


-- Train the model!
num_train = train_len
num_iterations = max_epochs * num_train
optim_config = {learningRate = 0.01}
tra_loss = 0

model:training()

for i = 1 , num_iterations do
	local epoch = math.floor(i / num_train) + 1

	-- Check if we are at the end of an epoch
	if i % num_train == 0 then
		model:resetStates() -- Reset hidden states

		-- Maybe decay learning rate
		if epoch % lr_decay_every == 0 then
			local old_lr = optim_config.learningRate
			optim_config = {learningRate = old_lr * lr_decay_factor}
		end
	end

	_, loss = optim.adam(f, params, optim_config)

	tra_loss = tra_loss + loss[1]
	
	local float_epoch = i / num_train 
	local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
	local args = {msg, float_epoch, max_epochs, i, num_iterations, loss[1]}
	print(string.format(unpack(args)))

	-- Epoch end
	if i % (num_train) == 0 then
		
		model:resetStates() -- Reset hidden states

		-- Evaluate loss on the validation set
		
		model:evaluate()
		table.insert(train_loss_history, tra_loss/num_train)
		tra_loss = 0
--[[
		local val_loss = 0
		for j = 1, 1800 do
			x = valset[j]:view(1,T,D):type(dtype)
			
			if j<=900 then y = torch.Tensor(T):fill(1)
			elseif j>900 then y = torch.Tensor(T):fill(2)
			end

			local scores = model:forward(x)
			scores_view = scores:view(T ,-1)
			val_loss = val_loss + crit:forward(scores_view, y)
		end
		
		val_loss = val_loss / 1800
		print('val_loss = ', val_loss)
		table.insert(val_loss_history, val_loss)
]]--
		err_r = err_test()
		print(err_r)
		table.insert(err_rate, err_r)
		
		model:resetStates() -- Reset hidden states

		model:training()
	end
end

plotT = torch.Tensor(max_epochs)

for k,v in ipairs(train_loss_history) do
	plotT[k]=v
end
                                    
plotV = torch.Tensor(max_epochs)

for k,v in ipairs(val_loss_history) do
	plotV[k]=v
end

plotE = torch.Tensor(max_epochs)
for k,v in ipairs(err_rate) do
	plotE[k]=v
end
gnuplot.plot({'train-loss',plotT},{'err-rate',plotE})


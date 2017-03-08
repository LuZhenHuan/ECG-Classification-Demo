require 'torch'
require 'nn'
require 'gnuplot'
require 'optim'
require 'VanillaRNN'
require 'EcgModelSimple'
require 'cutorch'
require 'cunn'

dtype = 'torch.CudaTensor'
--build a simple rnn 100 -> 100 -> 1
model = nn.EcgModelSimple(400,100,2):type(dtype)
crit = nn.CrossEntropyCriterion():type(dtype)


--Set up some variables we will use below

N, T = 50, 400		--opt.batch_size, opt.seq_length	
count = 1
max_epochs = 20 
train_loss_history = {}
err_rate = {}
err_sample_tf = {}
err_sample_ft = {}

params, grad_params = model:getParameters()

trainTemp = torch.load('RnnTrain1D.t7')
trainset = trainTemp:view(N,-1,400):transpose(1,2):clone()
train_len = trainset:size(1)
testset = torch.load('RnnTest.t7')

--load a batch
local function next_batch()
	
	x = trainset[count]:view(1,-1,T)
	count = count + 1
	
	if count == train_len then
		count = 1
    end

	y = torch.Tensor(N):fill(1)
	y[{{N/2+1,N}}] = 2
	
	x, y = x:type(dtype), y:type(dtype)

	return x,y
end

--test
local function err_test()
	err = 0
	model:evaluate()

	for i = 1, 39000 do

		x = testset[i]:view(1,1,400):type(dtype)

		a = model:forward(x):view(-1)
		if i <=19500 and a[1]<a[2] then
			err = err + 1
			table.insert(err_sample_tf, x)
		elseif i > 19500 and a[1]>a[2] then
			err = err + 1
			table.insert(err_sample_ft, x)
		end
		err_r = err/39000
	end
	model:training()
	return err_r
end


--loss function 
local function f(w)

	assert(w == params)
	grad_params:zero()
	--data_process()
	x,y = next_batch()

	local scores = model:forward(x)
	local loss = crit:forward(scores, y)
	local grad_scores = crit:backward(scores, y)
	model:backward(x, grad_scores)
	--model:resetStates()

	grad_params:clamp(-5, 5)

  return loss, grad_params
end


-- Train the model!
num_train = train_len
num_iterations = max_epochs * num_train
plot = torch.Tensor(10)
optim_config = {learningRate = 0.001}
model:training()

for i = 1 , num_iterations do
	
	_, loss = optim.adam(f, params, optim_config)
	table.insert(train_loss_history, loss[1])
	

	if i % num_train == 0 then
    	model:resetStates() -- Reset hidden states
	end
	
	local float_epoch = i / num_train 
	local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
	local args = {msg, float_epoch, max_epochs, i, num_iterations, loss[1]}
	print(string.format(unpack(args)))

	--[[if float_epoch%20 == 0 then
		err_r = err_test()
		print(err_r)
		table.insert(err_rate, err_r)
	end]]--


end

torch.save('EMS20.t7',model)

err_r = err_test()
print(err_r)
--table.insert(err_rate, err_r)

--[[plot = torch.Tensor(max_epochs/2)
for k, v in ipairs(err_rate) do
	plot[k] = v
end
gnuplot.plot(plot)]]--

--修正minibatch和BPTT

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
model = nn.EcgModelSimple(200,50,2):type(dtype)
crit = nn.CrossEntropyCriterion():type(dtype)


--Set up some variables we will use below

N, T ,D= 10, 10, 200	--opt.batch_size, opt.seq_length , word_dim	
count = 1
max_epochs = 300
train_loss_history = {}
val_loss_history = {}
err_rate = {}
err_sample_tf = {}
err_sample_ft = {}

params, grad_params = model:getParameters()

--data process
trainTemp = torch.load('RnnTrain1D.t7')
trainset = trainTemp:view(N,-1,T*D):transpose(1,2):clone()
train_len = trainset:size(1)

testTemp = torch.load('RnnTestCut.t7')
testset = testTemp:view(-1,T*D)
m = testset:size(1)

valset = torch.load('RnnVal.t7'):view(N,-1,T*D):transpose(1,2):clone()

--load a batch
local function next_batch(dataset)

	data_len = dataset:size(1)

	x = dataset[count]:view(N,-1,D)
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

		x = testset[i]:view(1,10,200):type(dtype)

		a = model:forward(x):view(-1)
		if i <=m/2 and a[19]<a[20] then
			err = err + 1
			table.insert(err_sample_tf, x)
			print(i)
		elseif i > m/2 and a[19]>a[20] then
			err = err + 1
			table.insert(err_sample_ft, x)
			print(i)
		end
		err_r = err
		
	end
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
tra_loss = 0

for i = 1 , num_iterations do
	
	_, loss = optim.adam(f, params, optim_config)
	tra_loss = tra_loss + loss[1]
	
	local float_epoch = i / num_train 
	local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
	local args = {msg, float_epoch, max_epochs, i, num_iterations, loss[1]}
	print(string.format(unpack(args)))
	
	-- Epoch end
	if i % num_train == 0 then
		
		-- Evaluate loss on the validation set
		
		--model:evaluate()
    	model:resetStates() -- Reset hidden states
		table.insert(train_loss_history, tra_loss/num_train)
		tra_loss = 0
--[[
		local val_loss = 0
		for j = 1, 180 do
			local x,y = next_batch(valset)
		
			local scores = model:forward(x)
			scores_view = scores:view(N*T ,-1)
			val_loss = val_loss + crit:forward(scores_view, y)
		end
		
		val_loss = val_loss / 180
		print('val_loss = ', val_loss)
		table.insert(val_loss_history, val_loss)

		err_r = err_test()
		print(err_r)
		table.insert(err_rate, err_r)
]]--	

		model:resetStates() -- Reset hidden states

		model:training()

	end

end

--torch.save('EMSs10*200e200.t7',model)		--s = sequence  e = epoch

plotT = torch.Tensor(max_epochs)

for k,v in ipairs(train_loss_history) do
	plotT[k]=v
end

 --[[                                       
plotV = torch.Tensor(max_epochs)

for k,v in ipairs(val_loss_history) do
	plotV[k]=v
end

gnuplot.plot({'train_loss',plotT},{'val_loss',plotV})


plotE = torch.Tensor(max_epochs)
for k,v in ipairs(err_rate) do
	plotE[k]=v
end
gnuplot.plot({'train_loss',plotT},{'val_loss',plotV},{'err_rate',plotE})

]]--
gnuplot.plot({'train_loss',plotT})

require 'torch'
require 'nn'
require 'gnuplot'
require 'optim'
require 'cutorch'
require 'cunn'
require 'VanillaRNN'
require 'EcgModelSimple'


dtype = 'torch.CudaTensor'
--build a simple rnn 100 -> 100 -> 1
model = nn.EcgModelSimple(400,100,2):type(dtype)
crit = nn.CrossEntropyCriterion():type(dtype)

--Set up some variables we will use below

N, T = 1, 400		--opt.batch_size, opt.seq_length	
count = 1
max_epochs = 20
train_loss_history = {}

params, grad_params = model:getParameters()

trainTemp = torch.load('RnnTrain.t7')
trainset = trainTemp:view(50,-1,400):transpose(1,2):clone()
--load a batch
function next_batch()	
	x = trainset[count]:view(1,-1,T)
	count = count + 1

	if count <= 50250 then
			y = torch.Tensor{1}
		else
			y = torch.Tensor{1}
	end

	if count == 110501 then
		count = 1
	end

	x, y = x:type(dtype), y:type(dtype)

	return x,y
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
num_train = 110500
num_iterations = max_epochs * num_train

optim_config = {learningRate = 0.01}
model:training()

for i = 1 , num_iterations do
	
	_, loss = optim.adam(f, params, optim_config)
	table.insert(train_loss_history, loss[1])
	
	local float_epoch = i / num_train + 1
	local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
	local args = {msg, float_epoch, max_epochs, i, num_iterations, loss[1]}
	print(string.format(unpack(args)))
			
end

torch.save('EMS20.t7',model)

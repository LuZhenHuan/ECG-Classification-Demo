require 'torch'
require 'nn'

require 'optim'
require 'VanillaRNN' 
require 'EcgModel'


dtype = 'torch.FloatTensor'
--build a simple rnn 100 -> 100 -> 1
model = nn.EcgModel(100,100,1):type(dtype)
crit = nn.MSECriterion():type(dtype)

--Set up some variables we will use below

N, T = 1, 100		--opt.batch_size, opt.seq_length	
count = 1
flag  = 1
epoch = 10
max_epochs = 10
train_loss_history = {}

params, grad_params = model:getParameters()

trainset = torch.load('RnnTrain.t7')
--load and process data
function data_process()

	
	xTemp = trainset[count]:view(4,-1,T)
	count = count + 1

	if count <= 50250 then
		y = torch.Tensor{0}:view(1,1,1)
	else
		y = torch.Tensor{1}:view(1,1,1)
	end
	
	if count == 110501 then
		count = 1
	end
end

--load a batch
function next_batch()
	
	
	x = xTemp[flag]:view(1,-1,100)
	flag = flag + 1
	if flag == 4 then
		flag = 1
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
  return loss, grad_params
end


-- Train the model!

num_train = 110500 * 4
num_iterations = epoch * num_train


optim_config = {learningRate = 0.01}
model:training()

for i = 1 , num_iterations do

	if i%4~=0 then
		x,y = next_batch()
		model:forward(x)
	else
		_, loss = optim.adam(f, params, optim_config)
		table.insert(train_loss_history, loss[1])
		--print(loss)
		data_process()

		model:resetStates()

		local float_epoch = i / num_train + 1
    	local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
    	local args = {msg, float_epoch, max_epochs, i, num_iterations, loss[1]}
    	print(string.format(unpack(args)))
	end
	
end






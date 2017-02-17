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
params, grad_params = model:getParameters()


--load and process data
function data_process()

	trainset = torch.load('RnnTrain.t7')
	xTemp = trainset[count]:view(4,-1,T)
	count = count + 1

	if count <= 50250 then
		y = torch.Tensor{0}:view(1,1,1)
	else
		y = torch.Tensor{1}:view(1,1,1)
	end
end

--读取下一个batch
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
  
  --grad_params:zero()
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
optim_config = {learningRate = 0.01}
model:training()

for i = 1 ,20 do

	if i%4~=0 then
		x,y = next_batch()
		model:forward(x)
	else
		_, loss = optim.adam(f, params, optim_config)
		print(loss)
		data_process()

		--model:resetStates()

	--end
	
end






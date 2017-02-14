require 'torch'
require 'nn'

require 'optim'
require 'VanillaRNN' 
require 'EcgModel'


dtype = 'torch.FloatTensor'
--build a simple rnn 
model = nn.EcgModel(2,4,2):type(dtype)
crit = nn.MSECriterion():type(dtype)

--Set up some variables we will use below
dtype = 'torch.FloatTensor'

N, T = 1, 2		--opt.batch_size, opt.seq_length	
flag  = 1
params, grad_params = model:getParameters()


--data
trainset = torch.Tensor{1,2,3,4,5,6}:view(N,-1,T):transpose(1,2):type(dtype)

--读取下一个batch
function next_batch()
	x = trainset[flag]:view(1,-1,T)
	flag = flag + 1
	if flag == 4 then
		flag = 1
	end
	
	return x
end


--loss function 
local function f(w)
  
  grad_params:zero()
  x = next_batch()
  y = torch.Tensor{0,1}
  x, y = x:type(dtype), y:type(dtype)
  
  local scores = model:forward(x)
  local loss = crit:forward(scores, y)
  local grad_scores = crit:backward(scores, y)
  model:backward(x, grad_scores)
  model:resetStates()
  return loss, grad_params
end


-- Train the model!
optim_config = {learningRate = 0.01}
model:training()

for i = 1 ,20 do
	
	if i%3~=0 then
		x = next_batch()
		model:forward(x)
	else
		_, loss = optim.adam(f, params, optim_config)
		print(loss)
		model:resetStates()

	end
	
end

model:evaluate()
model:resetStates()
model:forward(torch.Tensor{1,2}:type(dtype):view(1,1,-1))
model:forward(torch.Tensor{3,4}:type(dtype):view(1,1,-1))
result = model:forward(torch.Tensor{5,6}:type(dtype):view(1,1,-1))
print(result)





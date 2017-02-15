require 'torch'
require 'nn'
require 'optim'
--require 'VanillaRNN'
--require 'EcgModel'

--Set up some variables we will use below
dtype = 'torch.FloatTensor'
N, T = 2, 2		--opt.batch_size, opt.seq_length	
flag  = 1
params, grad_params = model:getParameters()

--build a simple rnn  2，4，2
model = nn.EcgModel(2,4,2):type(dtype)
crit = nn.ClassNLLCriterion():type(dtype)

--load data
x = torch.Tensor{1,2,3,4,5,6}:view(3,-1,2)
y = torch.Tensor{0,1}



--loss function 
local function f(w)
  
  grad_params:zero()
  
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

for i = 1 ,100 do
	
	if i%3~=0 then
		model:forward(x)
	else
		_, loss = optim.adam(f, params, optim_config)
		
	end
	
end

model:evaluate()
--model:resetStates()

--[[print(model:forward(torch.Tensor{1,1}:type(dtype):view(1,1,-1)))
print(model:forward(torch.Tensor{2,2}:type(dtype):view(1,1,-1)))
print(model:forward(torch.Tensor{3,3}:type(dtype):view(1,1,-1)))
]]-- 

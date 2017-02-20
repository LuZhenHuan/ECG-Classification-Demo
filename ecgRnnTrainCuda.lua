require 'torch'
require 'nn'
require 'gnuplot'
require 'optim'
require 'cutorch'
require 'cunn'
require 'VanillaRNN' 
require 'EcgModel'


dtype = 'torch.CudaTensor'
--build a simple rnn 100 -> 100 -> 1
model = nn.EcgModel(400,100,1):type(dtype)
crit = nn.MSECriterion():type(dtype)

--Set up some variables we will use below

N, T = 1, 400		--opt.batch_size, opt.seq_length	
count = 1
flag  = 1
max_epochs = 1
train_loss_history = {}

params, grad_params = model:getParameters()

trainset = torch.load('RnnTrain.t7')
--load and process data
function data_process(dataset)

	
	xTemp = dataset[count]:view(1,-1,T)
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
	
	x = xTemp[1]:view(1,-1,100)
	
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
num_iterations = max_epochs * num_train

optim_config = {learningRate = 0.01}
model:training()

for i = 1 , num_iterations do
	data_process(trainset)
	if i%4~=0 then
		x,y = next_batch()
		model:forward(x)
	else
		_, loss = optim.adam(f, params, optim_config)
		--table.insert(train_loss_history, loss[1])
		--print(loss)
		

		model:resetStates()

		local float_epoch = i / num_train + 1
    	local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
    	local args = {msg, float_epoch, max_epochs, i, num_iterations, loss[1]}
    	print(string.format(unpack(args)))
	end
	
end

--let's test the model
result = 0
testset = torch.load('RnnTest.t7')
count = 1
flag = 1
erTemp = 0
plota = torch.Tensor(30000) 
data_process(testset)
model:resetStates()
for i = 1 , 152000 do

	x,y = next_batch()
	a = model:forward(x):view(1)
	if i%4 == 0 then
		plota[i/4]= a[1]

		model:resetStates()
		if i <= 78000 and a[1]>0.5 then erTemp = erTemp + 1
		elseif i > 78000 and a[1]<0.5 then erTemp = erTemp + 1
		end
		data_process(testset)
	end
end
gnuplot.plot(plota)
print(erTemp)


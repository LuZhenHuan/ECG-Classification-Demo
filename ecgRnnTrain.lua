require 'torch'
require 'nn'
require 'gnuplot'
require 'optim'
require 'VanillaRNN' 
require 'EcgModel'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_data', 'data/RnnTrain.t7')
cmd:option('-batch_size', 50)
cmd:option('-seq_length', 50)

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'rnn')
cmd:option('-input_size', 64)
cmd:option('-rnn_size', 128)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0)
cmd:option('-batchnorm', 0)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 2e-3)
cmd:option('-grad_clip', 5)
cmd:option('-lr_decay_every', 5)
cmd:option('-lr_decay_factor', 0.5)

-- Output options
cmd:option('-print_every', 1)
cmd:option('-checkpoint_every', 1000)
cmd:option('-checkpoint_name', 'cv/checkpoint')

-- Benchmark options
cmd:option('-speed_benchmark', 0)
cmd:option('-memory_benchmark', 0)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')

local opt = cmd:parse(arg)

-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
else
  -- Memory benchmarking is only supported in CUDA mode
  opt.memory_benchmark = 0
  print 'Running in CPU mode'
end

-- Initialize the DataLoader and vocabulary
local loader = DataLoader(opt)

-- Initialize the model and criterion
local opt_clone = torch.deserialize(torch.serialize(opt))

local model = nil
local start_i = 0
if opt.init_from ~= '' then
  print('Initializing from ', opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  model = checkpoint.model:type(dtype)
  if opt.reset_iterations == 0 then
    start_i = checkpoint.i
  end
else
  model = nn.EcgModel(opt_clone):type(dtype)
end
local params, grad_params = model:getParameters()
local crit = nn.CrossEntropyCriterion():type(dtype)


--Set up some variables we will use below

N, T = 1, 100		--opt.batch_size, opt.seq_length	
count = 1
flag  = 1
epoch = 1
max_epochs = 1 
train_loss_history = {}

params, grad_params = model:getParameters()

trainset = torch.load('RnnTrain.t7')
--load and process data
function data_process(dataset)

	
	xTemp = dataset[count]:view(4,-1,T)
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
data_process(trainset)

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
num_iterations = max_epochs * num_train

optim_config = {learningRate = 0.01}
model:training()

for i = 1 , num_iterations do

	if i%4~=0 then
		x,y = next_batch()
		model:forward(x)
	else
		_, loss = optim.adam(f, params, optim_config)
		--table.insert(train_loss_history, loss[1])
		--print(loss)
		data_process(trainset)

	--	model:resetStates()

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
for i = 1 , 120000 do

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


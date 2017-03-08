require 'VanillaRNN'
require 'nn'
require 'EcgModelSimple'
require 'cunn'
require 'cutorch'

--let's test the model
model = torch.load('EMS20.t7')
dtype = 'torch.CudaTensor'


testset = torch.load('RnnTest.t7')
--count = 1
--flag = 1
--erTemp = 0
err = 0
plot = torch.Tensor(39000)
--view2 = nn.View(1, -1)
--data_process(testset)
err_rate = {}
err_sample_tf = {}
err_sample_ft = {}

local function err_test()
	err = 0
	model:resetStates()
	model:evaluate()

	for i = 1, 39000 do

		x = testset[i]:view(1,1,400):type(dtype)

		a = model:forward(x):view(-1)
		if i <=1000 and a[1]<a[2] then
			err = err + 1
			table.insert(err_sample_tf, x)
		elseif i > 19500 and a[1]>a[2] then
			err = err + 1
			table.insert(err_sample_ft, x)
		end
		err_r = err
		
	end
	model:training()
	return err_r
end

err_r = err_test()
print(err_r)
--table.insert(err_rate, err_r)

--[[plot = torch.Tensor(max_epochs/2)
for k, v in ipairs(err_rate) do
	plot[k] = v
end
gnuplot.plot(plot)]]--

print(model:forward(err_sample_tf[1]))

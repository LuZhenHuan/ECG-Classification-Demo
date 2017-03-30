require 'VanillaRNN'
require 'nn'
require 'EcgModelSimple'
require 'cunn'
require 'cutorch'

--let's test the model
model = torch.load('EMSs10*200e200.t7')
dtype = 'torch.CudaTensor'

N, T ,D= 10, 10, 200

testTemp = torch.load('RnnTestCut.t7')
testset = testTemp:view(-1,T*D)

m = testset:size(1)
--count = 1
--flag = 1
--erTemp = 0
err = 0
plot = torch.Tensor(m)
--view2 = nn.View(1, -1)
--data_process(testset)
err_rate = {}
err_sample_tf = {}
err_sample_ft = {}

local function err_test()
	err = 0
	model:resetStates()
	model:evaluate()

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

--print(model:forward(err_sample_tf[1]))

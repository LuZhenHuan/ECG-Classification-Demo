require 'VanillaRNN'
require 'nn'
require 'EcgModelSimple'
require 'cunn'
require 'cutorch'

--let's test the model
model = torch.load('EMSTR20.t7')
dtype = 'torch.CudaTensor'

T ,D= 10, 200

testTemp = torch.load('RnnTestCut.t7')
testset = testTemp:view(-1,T*D)

m = testset:size(1)

err = 0
plot = torch.Tensor(m)

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
		
		elseif i > m/2 and a[19]>a[20] then
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

--print(model:forward(err_sample_tf[1]))

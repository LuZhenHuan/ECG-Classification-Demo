require 'VanillaRNN'
require 'nn'
require 'EcgModelSimple'
require 'cunn'
require 'cutorch'

--let's test the model
model = torch.load('EMSTRTest0.007.t7')
dtype = 'torch.CudaTensor'

N, T ,D= 20, 5, 400	--opt.batch_size, opt.seq_length , word_dim	

testTemp = torch.load('RnnTestCut.t7')/4
testset = testTemp:view(-1,T*D):type(dtype)

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

		x = testset[i]:view(1,T,D):type(dtype)

		a = model:forward(x):view(-1)
		if i <= m/2 and a[T*2-1]<a[2*T] then
			err = err + 1
			table.insert(err_sample_tf, x)
			--print(i)
		elseif i > m/2 and a[T*2-1]>a[2*T] then
			err = err + 1
			table.insert(err_sample_ft, x)
			--print(i)
		end
	end
	err_r = err/m
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

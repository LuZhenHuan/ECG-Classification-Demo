require 'VanillaRNN'
require 'nn'
require 'EcgModelSimple'
require 'cunn'
require 'cutorch'

--let's test the model
model = torch.load('EMS20.t7')
dtype = 'torch.FloatTensor'


testset = torch.load('RnnTest.t7')
--count = 1
--flag = 1
--erTemp = 0
err = 0
plot = torch.Tensor(39000)
model:evaluate()
view2 = nn.View(1, -1)
--data_process(testset)

for i = 1, 39000 do

	x = testset[i]:view(1,1,400):type(dtype)

	a = model:forward(x):view(-1)
	if i <=39000 and a[1]<a[2] then
		err = err + 1
	plot[i]=a
	end
	

end
print(err)


--[[for i = 1 , 152000 do

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
print(erTemp)]]--


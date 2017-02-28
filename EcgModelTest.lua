--let's test the model
model = torch.load('ecgmodelsimple5epoch.t7')


result = 0
testset = torch.load('RnnTest.t7')
count = 1
flag = 1
erTemp = 0
plota = torch.Tensor(100) 
model:evaluate()

data_process(testset)
model:resetStates()

for i = 1, 10000 do
	x,y = next_batch()
	a = model:forward(x):view(-1)
	if i%4 == 0 then
		print(a)

		model:resetStates()
	
		data_process(testset)
	end
end


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


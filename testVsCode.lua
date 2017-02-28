count = 1
trainset = torch.load('RnnTrain.t7')
--load and process data
function data_process(dataset)

	xTemp = dataset[count]:view(-1,T)
	count = count + 1

	if count <= 50250 then
		y = torch.Tensor{0,1}:view(1,1,-1)
	else
		y = torch.Tensor{1,0}:view(1,1,-1)
	end
	
	if count == 110501 then
		count = 1
	end
end

print(#trainset)
data_process(trainset)
print(#xTemp)
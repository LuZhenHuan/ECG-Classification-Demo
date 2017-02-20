--[[require 'torch'
csvFile = io.open('RnnTest.csv', 'r')  
 header = csvFile:read()

 data = torch.Tensor()

 i = 0  
for line in csvFile:lines('*l') do  
  i = i + 1
   l = line:split(',')
  for key, val in ipairs(l) do
    data[i][key] = val
  end
end

csvFile:close() 

print(#data)
--print (data)
]]--

require 'csvigo'
dataOriginal = csvigo.load{path = 'RnnTrain.csv',mode = 'raw'}
M, N = 110500,400

Data = torch.Tensor(M,N)

for i=1,M do
	for j=1,N do
		Data[i][j] = dataOriginal[i][j];
	end
end
print(#Data)	
--print (Data)	
torch.save('RnnTrain.t7',Data)

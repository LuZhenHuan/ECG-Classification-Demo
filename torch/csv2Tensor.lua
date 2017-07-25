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
dataOriginal = csvigo.load{path = 'D1Test.csv',mode = 'raw'}
M, N = 6500000,400

Data = torch.Tensor(M)

for i=1,M do
	
		Data[i] = dataOriginal[i];

end
print(#Data)	
--print (Data)	
torch.save('D1Test.t7',Data)

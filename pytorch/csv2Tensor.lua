require 'torch'
csvFile = io.open('D1Test.csv', 'r')  
-- header = csvFile:read()

j = 6500000

data = torch.Tensor(j)

i = 0  

for line in csvFile:lines() do  
  i = i + 1
  data[i] = line

end

--[[
for line in csvFile:lines() do  
  i = i + 1
   l = line:split(',')
  for key, val in ipairs(l) do
    data[i][key] = val
  end
end
]]--

csvFile:close() 
torch.save('D8Test.t7',data)

print(#data)
--print (data)

--[[
require 'csvigo'
dataOriginal = csvigo.load{path = 'D1Train.csv',mode = 'raw'}
M, N = 6500000,400

Data = torch.Tensor(M)

for i=1,M do
	
		Data[i] = dataOriginal[i][1];

end
print(#Data)	
--print (Data)	
torch.save('D1Train.t7',Data)
]]--
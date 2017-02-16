local csvFile = io.open('testTableTrain.csv', 'r')  
local header = csvFile:read()

local data = torch.Tensor()

local i = 0  
for line in csvFile:lines('*l') do  
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
    data[i][key] = val
  end
end

csvFile:close() 

print(#data)
print (data)


--[[
require 'csvigo'
dataOriginal = csvigo.load{path = '112233.csv',mode = 'raw'}
M, N = 13, 6

Data = torch.Tensor(M,N)

for i=1,M do
	for j=1,N do
		Data[i][j] = dataOriginal[i][j];
	end
end
print(#Data)	
print (Data)	]]--

require 'torch'
require 'hdf5'

N, T = 1, 1

-- Just slurp all the data into memory
splits = {}
f = hdf5.open('/home/lu/torch-rnn/my_data.h5', 'r')

splits.train = f:read('/train'):all()
splits.val = f:read('/val'):all()
splits.test = f:read('/test'):all()

data.x_splits = {}
data.y_splits = {}
data.split_sizes = {}

for split, v in pairs(splits) do
	local num = v:nElement()
	local extra = num % (N * T)

	-- Ensure that `vy` is non-empty
	if extra == 0 then
	  extra = N * T
	end

	-- Chop out the extra bits at the end to make it evenly divide
	local vx = v[{{1, num - extra}}]:view(N, -1, T):transpose(1, 2):clone()
	local vy = v[{{2, num - extra + 1}}]:view(N, -1, T):transpose(1, 2):clone()
	
	data.x_splits[split] = vx
	data.y_splits[split] = vy
	data.split_sizes[split] = vx:size(1)
end

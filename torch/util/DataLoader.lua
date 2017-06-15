require 'torch'
 

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoader')


function DataLoader:__init(kwargs)
  local trainset = torch.load('RnnTrain1Dcut.t7')
  local valset = torch.load('RnnVal1Dcut.t7')
  local testset = torch.load('RnnTest.t7'):view(-1)

  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.seq_length = utils.get_kwarg(kwargs, 'seq_length')
  local N, T = self.batch_size, self.seq_length

  -- Just slurp all the data into memory
  local splits = {}
  splits.train = trainset
  splits.val = valset
  splits.test = testset

  self.x_splits = {}
  self.y_splits = {}
  self.split_sizes = {}
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

    self.x_splits[split] = vx
    self.y_splits[split] = vy
    self.split_sizes[split] = vx:size(1)
  end

  self.split_idxs = {train=1, val=1, test=1}
end


function DataLoader:nextBatch(split)
  local idx = self.split_idxs[split]
  assert(idx, 'invalid split ' .. split)
  local x = self.x_splits[split][idx]
  local y = self.y_splits[split][idx]
  if idx == self.split_sizes[split] then
    self.split_idxs[split] = 1
  else
    self.split_idxs[split] = idx + 1
  end
  x = x:view(1,50,100)
  return x, y
end


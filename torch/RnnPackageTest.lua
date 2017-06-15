require 'rnn'

batchSize = 8
rho = 5
hiddenSize = 10
nIndex = 10000-- RNN
r = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

rnn = nn.Sequential()
rnn:add(nn.Sequencer(r))
rnn:add(nn.Sequencer(nn.Linear(hiddenSize, nIndex)))
rnn:add(nn.Sequencer(nn.LogSoftMax()))

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- dummy dataset (task is to predict next item, given previous)
sequence = torch.randperm(nIndex)

offsets = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()*batchSize))
end
offsets = torch.LongTensor(offsets)

lr = 0.1
i = 1 while true do
   -- prepare inputs and targets
   local inputs, targets = {},{}
   for step=1,rho do
      -- a batch of inputs
      table.insert(inputs, sequence:index(1, offsets))
      -- incement indices
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > nIndex then
            offsets[j] = 1
         end
      end
      -- a batch of targets
      table.insert(targets, sequence:index(1, offsets))
   end

   local outputs = rnn:forward(inputs)
   local err = criterion:forward(outputs, targets)
   print(i, err/rho)
   i = i + 1
   local gradOutputs = criterion:backward(outputs, targets)
   rnn:backward(inputs, gradOutputs)
   rnn:updateParameters(lr)
   rnn:zeroGradParameters()
end
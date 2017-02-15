require 'rnn'

batchSize = 1
rho = 3
hiddenSize = 20
nIndex = 10000-- RNN
r = nn.Recurrent(
   hiddenSize, nn.Linear(1, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

rnn = nn.Sequential()
rnn:add(r)
rnn:add(nn.Linear(hiddenSize, 1))
rnn:add(nn.Sigmoid())

criterion = nn.SequencerCriterion(nn.MSECriterion())  


lr = 0.1
updateInterval = 4
i = 1 
while true do
   -- a batch of inputs
   local input = sequence:index(1, offsets)
   local output = rnn:forward(input)
   -- incement indices
   offsets:add(1)
   for j=1,batchSize do
      if offsets[j] > nIndex then
         offsets[j] = 1
      end
   end
   
   local target = sequence:index(1, offsets)
   local err = criterion:forward(output, target)
   print(err)
   local gradOutput = criterion:backward(output, target)
   -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
   rnn:backward(input, gradOutput)

   i = i + 1
   -- note that updateInterval < rho
   if i % updateInterval == 0 then
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      -- 2. updates parameters
      rnn:backwardThroughTime()
      rnn:updateParameters(lr)
      rnn:zeroGradParameters()
   end
   end

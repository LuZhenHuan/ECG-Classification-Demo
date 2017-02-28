require 'torch'
require 'nn'

--require 'VanillaRNN'

local EM, parent = torch.class('nn.EcgModelSimple', 'nn.Module')

function EM:__init(input_dim, hidden_dim ,output_dim)
  
  local V, H ,Y= input_dim, hidden_dim ,output_dim      --input size, rnn_size, label size

  self.net = nn.Sequential()
  self.rnns = {}

  rnn = nn.VanillaRNN(V, H)
  
  rnn.remember_states = true
  table.insert(self.rnns, rnn)
  self.net:add(rnn)

	self.view1 = nn.View(1, 1, -1):setNumInputDims(3)     
	self.view2 = nn.View(1, -1):setNumInputDims(2)

	self.net:add(self.view1)
	self.net:add(nn.Linear(H, Y))
	self.net:add(self.view2)
end

function EM:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)

  return self.net:forward(input)
end


function EM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end


function EM:parameters()
  return self.net:parameters()
end


function EM:training()
  self.net:training()
  parent.training(self)
end


function EM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end


function EM:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end

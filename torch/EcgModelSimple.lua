require 'torch'
require 'nn'
require 'VanillaRNN'
require 'LSTM'

local EM, parent = torch.class('nn.EcgModelSimple', 'nn.Module')

function EM:__init(input_dim, hidden_dim ,output_dim)
  
  local V, H ,Y= input_dim, hidden_dim ,output_dim      --input size, rnn_size, label size

  self.net = nn.Sequential()
  self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}
  self.bn_view_in = {}
  self.bn_view_out = {}

  rnn = nn.LSTM(V, H)
  
  rnn.remember_states = true
  table.insert(self.rnns, rnn)
  self.net:add(rnn)
--[[
  local view_in = nn.View(1, 1, -1):setNumInputDims(3)
  table.insert(self.bn_view_in, view_in)
  self.net:add(view_in)
  self.net:add(nn.BatchNormalization(H))
  local view_out = nn.View(1, -1):setNumInputDims(2)
  table.insert(self.bn_view_out, view_out)
  self.net:add(view_out)

  self.net:add(nn.Dropout(0.2))
]]--
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
--[[
  for _, view_in in ipairs(self.bn_view_in) do
    view_in:resetSize(N * T, -1)
  end
  
  for _, view_out in ipairs(self.bn_view_out) do
    view_out:resetSize(N, T, -1)
  end
]]--
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

function EM:clearState()
  self.net:clearState()
end

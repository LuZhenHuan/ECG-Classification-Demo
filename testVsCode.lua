require 'torch'
require 'nn'
require 'gnuplot'
require 'optim'
require 'VanillaRNN'

model = nn.Sequential()
model:add(nn.View(1,2,-1))
model:add(nn.VanillaRNN(3,5))
--model:add(nn.View(-1))

--model:add(nn.Linear(3,2))
--model:add(nn.LogSoftMax(5,5))
--print(model)
--params, grad_params = model:getParameters()
crit = nn.CrossEntropyCriterion()

--model = nn.SoftMax()
x = torch.Tensor{1,2,3,4,5,6}
y = 1

--[[local function f(w)

    scores = model:forward(x)
    --model:zeroGradParameters()
    loss = crit:forward(scores, y)  
    grad_scores = crit:backward(scores,y)
    model:backward(x, grad_scores)
   -- model:updateParameters(0.02)
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  return loss, grad_params
end
for i = 1, 200 do

    _, loss = optim.adam(f, params, optim_config)
    print(loss)

end]]--

--scores = model:forward(x)
--loss = crit:forward(scores, y)  
model:evaluate()

print(model:forward(x))

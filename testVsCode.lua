require 'torch'
require 'nn'
require 'gnuplot'
require 'optim'
require 'VanillaRNN'

model = nn.Sequential()
--model:add(nn.View(1,50,100))
model:add(nn.VanillaRNN(100,100))
view1 = nn.View(50, -1)
view2 = nn.View(50,100, -1)
--model:add(nn.View(-1))
model:add(view1)
model:add(nn.Linear(100,2))
--model:add(view2)
--model:add(nn.LogSoftMax(5,5))
--print(model)
--params, grad_params = model:getParameters()
crit = nn.CrossEntropyCriterion()

--model = nn.SoftMax()
--x = torch.Tensor{1,2,3,4,5,6}
--y = 1
local trainset = torch.load('RnnTrain1D.t7')
vx = trainset:view(50, -1, 100):transpose(1, 2):clone()
x = vx[1]:view(1,50,100)
y = torch.Tensor(50):fill(1)
--[[local function f(w)

    scores = model:forward(x)
    model:zeroGradParameters()
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

end
  

for i = 1,10 do
    if i<=5 and 1<2 then
    print '1'
    elseif i >5 and 1>2 then
    print '2'
    end
end]]--

model:evaluate()

scores = model:forward(x)
loss = crit:forward(scores, y)
    grad_scores = crit:backward(scores,y)
    model:backward(x, grad_scores)




print(scores,loss,grad_scores)

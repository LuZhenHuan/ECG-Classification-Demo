require 'torch'
require 'nn'
require 'gnuplot'
require 'optim'
require 'VanillaRNN'

model = nn.Sequential()

model:add(nn.VanillaRNN(2,5))
model:add(nn.View(-1))
model:add(nn.Linear(5,1))

params, grad_params = model:getParameters()
crit = nn.CrossEntropyCriterion()

x = torch.Tensor(2):view(1,1,-1)
y = torch.Tensor{0,1}

print('x=',x,'y=',y)
scores = model:forward(x)
print(scores)

scores_view = scores:view(3, -1)
y_view = y:view(-1)
loss = crit:forward(scores_view, y_view)  

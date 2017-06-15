require 'torch'  
require 'nn'  
require 'optim'  
--require 'cunn'  
--require 'cutorch'  
mnist = require 'mnist'  
--mnist中包含我们需要的训练数据和测试数据  
fullset = mnist.traindataset()  
testset = mnist.testdataset()  
    
trainset = {  
    size = 50000,  
    data = fullset.data[{{1,50000}}]:double(),  
    label = fullset.label[{{1,50000}}]  
}  
    
validationset = {  
    size = 10000,  
    data = fullset.data[{{50001,60000}}]:double(),  
    label = fullset.label[{{50001,60000}}]  
}  
--数据减去他们的均值  
trainset.data = trainset.data - trainset.data:mean()  
validationset.data = validationset.data - validationset.data:mean()  
model = nn.Sequential()  
model:add(nn.Reshape(1, 28, 28))  
model:add(nn.MulConstant(1/256.0*3.2))  
model:add(nn.SpatialConvolutionMM(1, 20, 5, 5, 1, 1, 0, 0))  
model:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))  
model:add(nn.SpatialConvolutionMM(20, 50, 5, 5, 1, 1, 0, 0))  
model:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))  
model:add(nn.Reshape(4*4*50))  
model:add(nn.Linear(4*4*50, 500))  
model:add(nn.ReLU())  
model:add(nn.Linear(500, 10))  
model:add(nn.LogSoftMax())  
    
model = require('weight-init')(model, 'xavier')  
    
criterion = nn.ClassNLLCriterion()  
    
--model = model:cuda()  
--criterion = criterion:cuda()  
--trainset.data = trainset.data:cuda()  
--trainset.label = trainset.label:cuda()  
--validationset.data = validationset.data:cuda()  
--validationset.label = validationset.label:cuda()  
    
sgd_params = {  
    learningRate = 1e-2,  
    learningRateDecay = 1e-4,  
    weightDecay = 1e-3,  
    momentum = 1e-4  
}  
--获取模型的初始参数，x是权值，dl_dx是梯度  
x, dl_dx = model:getParameters()  
    
step = function(batch_size)  
    local current_loss = 0  
    local count = 0  
    local shuffle = torch.randperm(trainset.size)--随机下标，用来打乱训练数据  
    batch_size = batch_size or 200--批处理数目为200  
    for t = 1,trainset.size,batch_size do  
        -- setup inputs and targets for this mini-batch  
        local size = math.min(t + batch_size - 1, trainset.size) - t  
        local inputs = torch.Tensor(size, 28, 28)--:cuda()  
        local targets = torch.Tensor(size)--:cuda()  
        for i = 1,size do  
            local input = trainset.data[shuffle[i+t]]  
            local target = trainset.label[shuffle[i+t]]  
            -- if target == 0 then target = 10 end  
            inputs[i] = input--将随机打乱位置的数据集加入到训练输入和输出集  
            targets[i] = target  
        end  
        targets:add(1)  
        local feval = function(x_new)--这个函数是个迭代训练函数  
            -- reset data  
            if x ~= x_new then x:copy(x_new) end  
            dl_dx:zero()--梯度归零，类似这个函数zeroGradParameters()    
    
            -- perform mini-batch gradient descent  
            local loss = criterion:forward(model:forward(inputs), targets)--两次前向计算，一次是带入模型计算输出，一次是用输出和目标输出求误差  
            model:backward(inputs, criterion:backward(model.output, targets))--两次后向计算，一次是带入误差函数求梯度，一次是用梯度和输入数据更新权值  
    
            return loss, dl_dx--返回误差和梯度  
        end  
    
        _, fs = optim.sgd(feval, x, sgd_params)--优化函数  
    
        -- fs is a table containing value of the loss function  
        -- (just 1 value for the SGD optimization)  
        count = count + 1  
        current_loss = current_loss + fs[1]  
    end  
    
    -- normalize loss  
    return current_loss / count  
end  
    
eval = function(dataset, batch_size)  
    local count = 0  
    batch_size = batch_size or 200  
        
    for i = 1,dataset.size,batch_size do  
        local size = math.min(i + batch_size - 1, dataset.size) - i  
        local inputs = dataset.data[{{i,i+size-1}}]--:cuda()  
        local targets = dataset.label[{{i,i+size-1}}]:long()--:cuda()  
        local outputs = model:forward(inputs)--用训练好的模型计算输出  
        local _, indices = torch.max(outputs, 2)  
        indices:add(-1)  
        local guessed_right = indices:eq(targets):sum()--计算命中数目  
        count = count + guessed_right  
    end  
    
    return count / dataset.size  
end  
    
max_iters = 30  
    
do  
    local last_accuracy = 0  
    local decreasing = 0  
    local threshold = 1 -- how many deacreasing epochs we allow  
    for i = 1,max_iters do  
        local loss = step()  
        print(string.format('Epoch: %d Current loss: %4f', i, loss))  
        local accuracy = eval(validationset)  
        print(string.format('Accuracy on the validation set: %4f', accuracy))  
        if accuracy < last_accuracy then  
            if decreasing > threshold then break end  
            decreasing = decreasing + 1  
        else  
            decreasing = 0  
        end  
        last_accuracy = accuracy  
    end  
end  
    
testset.data = testset.data:double()  
eval(testset)  
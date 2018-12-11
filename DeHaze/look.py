import torch
import numpy as np
import scipy.io as sio
from collections import OrderedDict

model_params = torch.load('/home/lu/code/data/DeHaze_params_4.pkl')
print(model_params.keys())


#print(isinstance(model_params, OrderedDict))
weight = dict()


weight['conv1_w'] = model_params['conv1.weight'].cpu().numpy()
weight['maxout_w'] = model_params['maxout.weight'].cpu().numpy()
weight['conv2_w'] = model_params['conv2.weight'].cpu().numpy()
weight['conv3_w'] = model_params['conv3.weight'].cpu().numpy()
weight['conv4_w'] = model_params['conv4.weight'].cpu().numpy()
weight['conv5_w'] = model_params['conv5.weight'].cpu().numpy()
#weight['conv6_w'] = model_params['conv6.weight'].cpu().numpy()

weight['conv1_b'] = model_params['conv1.bias'].cpu().numpy()
weight['maxout_b'] = model_params['maxout.bias'].cpu().numpy()
weight['conv2_b'] = model_params['conv2.bias'].cpu().numpy()
weight['conv3_b'] = model_params['conv3.bias'].cpu().numpy()
weight['conv4_b'] = model_params['conv4.bias'].cpu().numpy()
weight['conv5_b'] = model_params['conv5.bias'].cpu().numpy()
#weight['conv6_b'] = model_params['conv6.bias'].cpu().numpy()

sio.savemat('/home/lu/code/data/DeHaze_params_4.mat', mdict = weight)

data = sio.loadmat('/home/lu/code/data/DeHaze_params_4.mat')
print(data['conv1_b'])

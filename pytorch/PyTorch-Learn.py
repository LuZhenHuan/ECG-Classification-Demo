# -*- coding: utf-8 -*-
#implenmenting a forward and backward passes through the network using numpy operations
import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

for i in range(1000):
    #forward pass
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    #compute loss
    loss = np.square(y_pred - y).sum()
    print(i, loss)

    #backprop
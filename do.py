# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:19:28 2022

@author: luigi
"""
import numpy as np
import random as r
from backyead import *
import matplotlib.pyplot as plt
import my_set
#AdaGrad
train_x,outs = my_set.data.load_data("nands")
#train_y = np.array(([1,0],[1,0],[1,0],[0,1]))
train_size = 4
in_size = 2
out_size = 2
loss_list = np.array(())
train_y = np.array(())

for i in range(len(outs)):
    data = [0,0]
    data[int(outs[i])] = 1
    train_y = np.append(train_y,data)
train_y = train_y.reshape(len(outs),out_size)

#train_y = outs
w = np.random.rand(out_size,in_size+1)
affin_layer = affin()
sigmoid_layer = sigmoid()
loss_layer = loss()
eta = 1.5
epoch = 1000
axis =  []
dw = np.copy(w)
dw = dw * 0
former_train_list = np.arange(0,len(outs),1)
def affins(x):
    global w,out_size,in_size
    aff_lay = affin()
    sig_lay = sigmoid()
    layer1 = aff_lay.do(w, x, out_size, 1)
    layer2 = sig_lay.do(layer1)
    return layer2
for i in range(epoch):
    axis += [i]
    x1 = np.random.choice(former_train_list,train_size)
    x = train_x[x1]
    y = train_y[x1]
    layer1 = affin_layer.do(w,x,out_size,train_size)
    layer1 = layer1.reshape(train_size,out_size)
    layer2 = sigmoid_layer.do(layer1)
    loss_list = np.append(loss_list,(loss_layer.do(y,layer2)))
    #ζの計算
    limloss = loss_layer.back()
    limsig = sigmoid_layer.back()
    zeta = limloss * limsig
    limaff = affin_layer.back()
    dw = np.array(())
    for i in range(len(zeta)):
        dw = np.append(dw,zeta[i]*limaff)
    dw = dw.reshape(out_size,in_size+1)
    h = dw * dw + 1    
    w -= dw * eta *(1/np.sqrt(h))
plt.plot(axis,loss_list)
plt.show()
layer1 = affin_layer.do(w,train_x,out_size,4)
layer1 = layer1.reshape(4,out_size)
layer2 = sigmoid_layer.do(layer1)
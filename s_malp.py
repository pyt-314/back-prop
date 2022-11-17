# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 10:08:43 2022

@author: luigi
"""

import numpy as np
import matplotlib.pyplot as plt
from backyead import *
import my_set

#データの準備
x_train,x_out = my_set.data.load_data("xors")
#入力サイズ、中間層のノードサイズ、出力サイズ
in_size = 2
node_size = 4
out_size = 2
#重みデータの準備
w1 = np.random.rand(node_size,in_size+1)
w2 = np.random.rand(out_size,node_size+1)
#各レイヤの準備
#affinレイヤ
l1 = affin()
l3 = affin()
#ruleレイヤ
l2 = rule()
#sigmoidレイヤ
l4 =sigmoid()
#loss(二乗誤差)
l5 = loss()
#one_hot_labelへの変換

x_test = []
for i in list(x_out):
    base = [0,0]
    base[int(i)] = 1
    x_test += ([base])
x_test = np.array((x_test))
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 08:37:23 2022

@author: luigi
"""

import numpy as np
import random as r
import matplotlib.pyplot as plt

class sigmoid:
    def __init__(self):
        self.grad = None
        self.out = None
    def do(self,x):
        self.x = x
        self.l = len(x)
        self.out = 1/(1+np.exp(-x))
        self.shapes_ = self.out.shape
        self.out1 = np.array(())
        for i in range(self.shapes_[1]):
            self.out1 = np.append(self.out1,np.sum(self.out[:,i]))
        #print(self.shapes_)
        return self.out
    def back(self):
        self.grad = self.out * (1 - self.out)
        self.grad = np.sum(self.grad,axis=0)
        return self.grad
class rule:
    def __init__(self):
        self.x = None
        self.grad = None
    def do(self,x):
        
        x[x<0] = 0
        self.x = np.copy(x)
        return x
    def back(self):
        self.grad = np.copy(self.x)
        self.grad[self.grad>0] = 1
        return self.grad
class affin:
    def __init__(self):
        self.node = None
    def do(self,w,x,node_size,in_size):
        self.node = node_size
        self.insize = in_size
        self.ins = x
        self.outs = np.array(())
        self.shape_ = x.shape
        for i in range(self.insize):
            a = np.array(())
            for j in range(self.node):
                
                a = np.append(a,np.dot(np.append(x[i],1),w[j]))
            self.outs = np.append(self.outs,a)
        return self.outs
    def back(self):
        ins_list = np.array(())
        for i in range(self.insize):
            ins_list = np.append(ins_list,np.append(self.ins[i],1))
        ins_list = ins_list.reshape(self.shape_[0],self.shape_[1]+1)
        self.ins_list = np.sum(ins_list,axis=0)
        return self.ins_list

class loss:
    def __init__(self):
        self.x = None
        self.y = None
    def do(self,y,x):
        self.x = x
        self.y = y
        return np.sum(((y-x) ** 2) / 2)
    def back(self):
        self.dx = np.sum(self.x - self.y,axis=0)
        return self.dx

class zeta:
    def last(lim_out,lim_loss):
        return lim_out * lim_loss
    def others(zeta,w_1,lim_out):
        zeta = zeta
        a = w_1.shape
        zata1 = np.array(())
        zata2 = np.array(())
        for i in range(len(zeta)):
            zata1 = np.append(zata1,np.sum(zeta[i]*w_1[i]))
        for i in range(len(zata1)):
            zata2 = np.append(zata2,zata1[i]*lim_out)
        zata2 = zata2.reshape(len(lim_out),len(zata1))
        zata2 = np.sum(zata2,axis=1)
        zata = zata2
        return zata
    def oth2(zeta_1,w_1,lim_out,in_size):
        kr_zata = np.dot(zeta_1,w_1)
        zata = np.array(())
        zata = lim_out*kr_zata[:-1]
        #zata = np.append(zata,kr_zata*in_size)
        return zata
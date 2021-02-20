# coding: utf-8
from common.np import *
from common.functions import *


class MSE:
    def __init__(self):
        self.cache=None
        self.params,self.grads=[],[]

    def forward(self,x,t):
        batch_size=x.shape[0]
        loss=0.5*np.sum((x-t)**2)/batch_size
        self.cache=(x,t)
        # print('loss',loss)
        return loss

    def backward(self,dout=1):
        x,t=self.cache
        batch_size=x.shape[0]
        dout/=batch_size
        dx=dout*(x-t)
        # print('dx',dx)
        return dx
        
        
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ
        self.params,self.grads=[],[]

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx    
        
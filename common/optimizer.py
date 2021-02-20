# coding: utf-8
from common.np import *

class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr

    def update(self,params,grads):
        for i in range(len(params)):
            params[i]-=self.lr*grads[i]


class Nesterov:
    '''
    Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)
    '''
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] *= self.momentum
            self.v[i] -= self.lr * grads[i]
            params[i] += self.momentum * self.momentum * self.v[i]
            params[i] -= (1 + self.momentum) * self.lr * grads[i]


class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param).astype('f'))
                self.v.append(np.zeros_like(param).astype('f'))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        if np.isnan(lr_t):
            print('lr_tがnanです')

        for i in range(len(params)):
            if np.count_nonzero(np.isinf(grads[i]**2))!=0:
                    print('grads[%i]**2にinfが含まれます'%i)
            if np.count_nonzero(np.isnan(self.m[i]))!=0:
                print('更新前のself.m[%i]にnanが含まれます'%i)
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            if np.count_nonzero(np.isnan(self.v[i]))!=0:
                print('更新前のself.v[%i]にnanが含まれます'%i)
            if np.count_nonzero(np.isinf(self.v[i]))!=0:
                print('更新前のself.v[%i]にinfが含まれます'%i)
                print(self.v[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            if np.count_nonzero(np.isnan(self.v[i]))!=0:
                print('更新後のself.v[%i]にnanが含まれます'%i)
                
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
            if np.count_nonzero(np.isnan(params[i]))!=0:
                print('params[%d]にnanが含まれます'%i)
# coding: utf-8
from common.np import *
from common.functions import *
from common.util import im2col, col2im


class Relu:
    def __init__(self):
        self.mask=None
        self.params,self.grads=[],[]
        
    def forward(self,x):
        self.mask=(x>0)
        out=x*self.mask
        return out
        
    def backward(self,dout):
        dx=dout*self.mask
        return dx


class Sigmoid:
    def __init__(self):
        self.params,self.grads=[],[]
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
        

class Affine:
    def __init__(self,W,b):
        self.params=[W,b]
        self.grads=[np.zeros_like(W),np.zeros_like(b)]
        self.x=None

    def forward(self,x):
        W,b=self.params
        out=np.dot(x,W)+b
        self.x=x
        return out

    def backward(self,dout):
        W,b=self.params
        db=np.sum(dout,axis=0)
        dx=np.dot(dout,W.T)
        dW=np.dot(self.x.T,dout)
        self.grads[0][...]=dW
        self.grads[1][...]=db
        return dx


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        self.params=[self.gamma,self.beta]
        self.grads=[np.zeros_like(self.gamma).astype('f'),np.zeros_like(self.beta).astype('f')]

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        self.grads[0][...]=self.dgamma
        self.grads[1][...]=self.dbeta
        
        return dx


class GAP:
    def __init__(self):
        self.cache=None
        self.params,self.grads=[],[]

    def forward(self,x):
        N,C,H,W=x.shape
        x=x.reshape(N,C,-1) #(N,C,H*W)
        out=np.mean(x,axis=-1)    #(N,C)
        self.cache=(N,C,H,W)

        return out

    def backward(self,dout):
        N,C,H,W=self.cache
        dout=1/(H*W)*dout
        dx=dout.reshape(N,C,1).repeat(H*W,axis=2)
        dx=dx.reshape(N,C,H,-1)

        return dx
        

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
        
        
class Tanh:
    def __init__(self):
        self.params,self.grads=[],[]
        self.out=None

    def forward(self,x):
        out=np.tanh(x)
        self.out=out
        # print(out)
        return out

    def backward(self,dout):
        dx=dout*(1.0-self.out)**2
        # print(dx)
        return dx
        

class Convolution:
    def __init__(self,W,b,pad,stride):
        self.params=[W,b]
        self.grads=[np.zeros_like(W),np.zeros_like(b)]
        self.pad=pad
        self.stride=stride
        self.x=None
        self.col=None
        self.col_W=None
        
    def forward(self,x):
        N,C,H,W=x.shape
        FN,C,FH,FW=self.params[0].shape
        
        out_h=(H+2*self.pad-FH)//self.stride+1
        out_w=(W+2*self.pad-FW)//self.stride+1
        
        col=im2col(x,FH,FW,self.stride,self.pad)    #col(N*out_h*out_w,C*FH*FW)
        col_W=self.params[0].reshape(FN,-1).T   #col_W(FN,C*FH*FW)
        
        out=np.dot(col,col_W)+self.params[1]    #dot((N*out_h*out_w,C*FH*FW),(C*FH*FW,FN))→(N*out_h*out_w,FN)
        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)    #(N*out_h*out_w,FN)→(N,FN,out_h,out_w)
        
        self.x=x
        self.col=col
        self.col_W=col_W
        
        return out
        
    def backward(self,dout):
        FN,C,FH,FW=self.params[0].shape
        dout=dout.transpose(0,2,3,1).reshape(-1,FN) #(N,FN,out_h,out_w)→(N*out_h*out_w,FN)
        
        db=np.sum(dout,axis=0)  #(N*out_h*out_w,FN)→(FN)
        dW=np.dot(self.col.T,dout)  #dot((C*FH*FW,N*out_h*out_w),(N*out_h*out_w,FN))→(C*FH*FW,FN)
        dW=dW.reshape(C,FH,FW,-1).transpose(3,0,1,2)    #(C*FH*FW,FN)→(FN,C,FH,FW)
        dx=np.dot(dout,self.col_W.T)    #dot((N*out_h*out_w,FN),(FN,C*FH*FW))→(N*out_h*out_w,C*FH*FW)
        dx=col2im(dx,self.x.shape,FH,FW,self.stride,self.pad)
        
        self.grads[0][...]=dW
        self.grads[1][...]=db
        
        return dx
        

class Pooling:
    def __init__(self,pool_h, pool_w, pad=0,stride=2):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.pad = pad
        self.stride = stride
        
        self.x = None
        self.arg_max = None
        self.params,self.grads=[],[]

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H +2*self.pad - self.pool_h) / self.stride)
        out_w = int(1 + (W +2*self.pad - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
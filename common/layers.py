# coding: utf-8
from common.np import *
from common.functions import *
from common.util import im2col, col2im,to_cpu,to_gpu
from common import layer_dictionary


class Relu:
    def __init__(self,input_shape,param_list=[]):
        self.mask=None
        self.params,self.grads=[],[]
        self.output_shape=input_shape
        
    def forward(self,x):
        self.mask=(x>0)
        out=x*self.mask
        return out
        
    def backward(self,dout):
        dx=dout*self.mask
        return dx


class Sigmoid:
    def __init__(self,input_shape,param_list=[]):
        self.params,self.grads=[],[]
        self.out = None
        self.output_shape=input_shape

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
        

class Affine:
    def __init__(self,input_shape,param_list):
        '''
        param_list=[node_num,weight_init_std]
        
        '''
        node_num,weight_init_std=param_list
        W=np.random.randn(input_shape[0],node_num)*np.sqrt(weight_init_std/node_num).astype('f')
        b=np.zeros(node_num).astype('f')
        self.params=[W,b]
        self.grads=[np.zeros_like(W).astype('f'),np.zeros_like(b).astype('f')]
        self.x=None
        self.output_shape=(node_num,)

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
    def __init__(self, input_shape, param_list=[0.9,None,None]):
        self.momentum,self.running_mean,self.running_var=param_list
        if len(input_shape)==1:
            pre_node_num=input_shape[0]
        else:
            pre_node_num=input_shape[0]*input_shape[1]*input_shape[2]
            
        self.gamma=np.ones(pre_node_num).astype('f')
        self.beta =np.zeros(pre_node_num).astype('f')
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        self.params=[self.gamma,self.beta]
        self.grads=[np.zeros_like(self.gamma).astype('f'),np.zeros_like(self.beta).astype('f')]
        self.output_shape=input_shape

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
    def __init__(self,input_shape,param_list=[]):
        self.cache=None
        self.params,self.grads=[],[]
        C,H,W=input_shape
        self.output_shape=(C,)

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
        
        
class Tanh:
    def __init__(self,input_shape,param_list=[]):
        self.params,self.grads=[],[]
        self.out=None
        self.output_shape=input_shape

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
    def __init__(self,input_shape,param_list):
        '''
        param_list=[filter_num,filter_h,filter_w,pad,stride,weight_init_std]
        
        '''
        C,H,W=input_shape
        FN,FH,FW,self.pad,self.stride,weight_init_std=param_list
        pre_node_num=C*FH*FW
        
        conv_W=np.random.randn(FN,C,FH,FW)*np.sqrt(weight_init_std/pre_node_num).astype('f')
        conv_b=np.zeros(FN).astype('f')
        
        self.params=[conv_W,conv_b]
        self.grads=[np.zeros_like(conv_W).astype('f'),np.zeros_like(conv_b).astype('f')]
        self.original_shape=None
        self.col=None
        self.col_W=None

        self.out_h=(H+2*self.pad-FH)//self.stride+1
        self.out_w=(W+2*self.pad-FW)//self.stride+1
        self.output_shape=(FN,self.out_h,self.out_w)
        
    def forward(self,x):
        self.original_shape=x.shape
        N,C,H,W=x.shape
        FN,C,FH,FW=self.params[0].shape
        
        self.col=im2col(x,FH,FW,self.stride,self.pad)    #col(N*out_h*out_w,C*FH*FW)
        self.col_W=self.params[0].reshape(FN,-1).T   #col_W(FN,C*FH*FW)
        
        out=np.dot(self.col,self.col_W)+self.params[1]    #dot((N*out_h*out_w,C*FH*FW),(C*FH*FW,FN))→(N*out_h*out_w,FN)
        out=out.reshape(N,self.out_h,self.out_w,-1).transpose(0,3,1,2)    #(N*out_h*out_w,FN)→(N,FN,out_h,out_w)
        return out
        
    def backward(self,dout):
        FN,C,FH,FW=self.params[0].shape
        dout=dout.transpose(0,2,3,1).reshape(-1,FN) #(N,FN,out_h,out_w)→(N*out_h*out_w,FN)
        
        db=np.sum(dout,axis=0)  #(N*out_h*out_w,FN)→(FN)
        dW=np.dot(self.col.T,dout)  #dot((C*FH*FW,N*out_h*out_w),(N*out_h*out_w,FN))→(C*FH*FW,FN)
        dW=dW.reshape(C,FH,FW,-1).transpose(3,0,1,2)    #(C*FH*FW,FN)→(FN,C,FH,FW)
        dx=np.dot(dout,self.col_W.T)    #dot((N*out_h*out_w,FN),(FN,C*FH*FW))→(N*out_h*out_w,C*FH*FW)
        dx=col2im(dx,self.original_shape,FH,FW,self.stride,self.pad)
        
        self.grads[0][...]=dW
        self.grads[1][...]=db
        
        return dx
        

class Pooling:
    def __init__(self,input_shape,param_list):
        '''
        param_list=[pool_h,pool_w,pad,stride]
        
        '''
        
        self.pool_h,self.pool_w,self.pad,self.stride = param_list
        
        self.original_shape=None
        self.arg_max = None
        self.params,self.grads=[],[]
        
        C,H,W=input_shape
        self.out_h = 1 + (H +2*self.pad - self.pool_h) // self.stride
        self.out_w = 1 + (W +2*self.pad - self.pool_w) // self.stride
        self.output_shape=(C,self.out_h,self.out_w)

    def forward(self, x):
        self.original_shape = x.shape
        N, C, H, W = x.shape

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, self.out_h, self.out_w, C).transpose(0, 3, 1, 2)

        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.original_shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
        
        
class ConvResNet:
    def __init__(self,input_shape,layer_list):
        '''
        input_dim=(C,H,W)
        layer_list=[
            ['conv',param_list],
            ['batchnorm',param_list],
            ['relu',params_list],
            ['pool',param_list]
            ]
        weight_init_std=2.0 if relu
                        1.0 if sigmoid
        '''
        layer_dict=layer_dictionary.layer_dict
            
        self.layers=[]
        for i,layer in enumerate(layer_list):
            self.layers.append(layer_dict[layer[0]](input_shape,layer[1]))
            input_shape=self.layers[i].output_shape
            
        self.params,self.grads=[],[]
        for layer in self.layers:
            self.params+=layer.params
            self.grads+=layer.grads
            
        self.x=None
        self.dx=None
        self.W=None
        self.col=None
        self.col_W=None
        self.output_shape=self.layers[-1].output_shape
        
    def forward(self,x,train_flg=True):
        self.x=x
        for layer in self.layers:
            if isinstance(layer,(BatchNormalization,ConvResNet)):
                x=layer.forward(x,train_flg)
            else:
                x=layer.forward(x)
                
        if self.x.shape==x.shape:
            out=x+self.x
        else:
            N,FN,out_h,out_w=x.shape
            N,C,H,W=self.x.shape
            if self.W is None:
                self.W=np.random.randn(FN,C).astype('f')
                self.params+=[self.W]
                self.grads+=[np.zeros_like(self.W).astype('f')]

                self.col=self.x.transpose(0,2,3,1).reshape(-1,C)  #(N*H*W,C)
                self.col_W=self.params[-1].T #(C,FN)
                out=np.dot(self.col,self.col_W).reshape(N,H,W,-1).transpose(0,3,1,2)  #(N*H*W,FN)→(N,FN,H,W)
                out+=x
            else:
                self.col=self.x.transpose(0,2,3,1).reshape(-1,C)  #(N*H*W,C)
                self.col_W=self.params[-1].T #(C,FN)
                out=np.dot(self.col,self.col_W).reshape(N,H,W,-1).transpose(0,3,1,2)  #(N*H*W,FN)→(N,FN,H,W)
                out+=x
        
        return out
        
    def backward(self,dout):
        self.dx=dout
        
        for layer in reversed(self.layers):
            dout=layer.backward(dout)
        
        if self.dx.shape==self.x.shape:
            dout+=self.dx
        else:
            N,FN,out_h,out_w=self.dx.shape
            N,C,H,W=self.x.shape
            dx_res=self.dx.transpose(0,2,3,1).reshape(N*H*W,-1)   #(N,FN,H,W)→(N*H*W,FN)
            dW=np.dot(self.col.T,dx_res)    #(C,FN)
            dW=dW.transpose()   #(FN,C)
            self.grads[-1][...]=dW
            dx_res=np.dot(dx_res,self.col_W.T)  #(N*H*W,C)
            dx_res=dx_res.reshape(N,H,W,-1).transpose(0,3,1,2)  #(N,C,H,W)
            dout+=dx_res
            
        return dout
        

class Flatten:
    def __init__(self,input_shape,param_list=[]):
        C,H,W=input_shape
        self.output_shape=(C*H*W,)
        self.original_shape=None
        self.params,self.grads=[],[]
        
    def forward(self,x):
        self.original_shape=x.shape
        x=reshape(x.shape[0],-1)
        return x
        
    def backward(self,dout):
        dout=dout.reshape(*self.original_shape)
        return dout
        
class Deconvolution:
    def __init__(self,input_shape,param_list):
        '''
        param_list=[filter_num,filter_h,filter_w,pad,stride,weight_init_std]
        
        '''
        C,H,W=input_shape
        FN,FH,FW,self.pad,self.stride,weight_init_std=param_list
        pre_node_num=C*FH*FW
        padded_h=(H+1)*self.pad+H
        padded_w=(W+1)*self.pad+W
        
        conv_W=np.random.randn(FN,C,FH,FW)*np.sqrt(weight_init_std/pre_node_num).astype('f')
        conv_b=np.zeros(FN).astype('f')
        
        self.params=[conv_W,conv_b]
        self.grads=[np.zeros_like(conv_W).astype('f'),np.zeros_like(conv_b).astype('f')]
        
        self.padded_x=None
        self.col=None
        self.col_W=None

        self.out_h=(padded_h-FH)//self.stride+1
        self.out_w=(padded_w-FW)//self.stride+1
        self.output_shape=(FN,self.out_h,self.out_w)
        self.mask=None
        self.original_shape=None

    def forward(self,x):
        self.original_shape=x.shape
        N,C,H,W=x.shape
        FN,C,FH,FW=self.params[0].shape
        
        idx_h=sorted(list(range(H+1))*self.pad)
        idx_w=sorted(list(range(W+1))*self.pad)
        self.mask=np.ones_like(x)
        if GPU:
            import numpy
            self.mask=to_cpu(self.mask)
            self.mask=numpy.insert(self.mask,idx_h,0,axis=2)
            self.mask=numpy.insert(self.mask,idx_w,0,axis=3)
            self.mask=self.mask.astype(bool)
            self.mask=to_gpu(self.mask)

            x=to_cpu(x)
            self.padded_x=numpy.insert(x,idx_h,0,axis=2)
            self.padded_x=numpy.insert(self.padded_x,idx_w,0,axis=3)
            self.padded_x=to_gpu(self.padded_x)
        else:
            self.mask=np.insert(self.mask,idx_h,0,axis=2)
            self.mask=np.insert(self.mask,idx_w,0,axis=3)
            self.mask=self.mask.astype(bool)
                
            self.padded_x=np.insert(x,idx_h,0,axis=2)
            self.padded_x=np.insert(self.padded_x,idx_w,0,axis=3)
        self.col=im2col(self.padded_x,FH,FW,self.stride,0)    #col(N*out_h*out_w,C*FH*FW)
        self.col_W=self.params[0].reshape(FN,-1).T   #col_W(FN,C*FH*FW)
        
        out=np.dot(self.col,self.col_W)+self.params[1]    #dot((N*out_h*out_w,C*FH*FW),(C*FH*FW,FN))→(N*out_h*out_w,FN)
        out=out.reshape(N,self.out_h,self.out_w,-1).transpose(0,3,1,2)    #(N*out_h*out_w,FN)→(N,FN,out_h,out_w)
        return out
        
    def backward(self,dout):
        FN,C,FH,FW=self.params[0].shape
        dout=dout.transpose(0,2,3,1).reshape(-1,FN) #(N,FN,out_h,out_w)→(N*out_h*out_w,FN)
        
        db=np.sum(dout,axis=0)  #(N*out_h*out_w,FN)→(FN)
        dW=np.dot(self.col.T,dout)  #dot((C*FH*FW,N*out_h*out_w),(N*out_h*out_w,FN))→(C*FH*FW,FN)
        dW=dW.reshape(C,FH,FW,-1).transpose(3,0,1,2)    #(C*FH*FW,FN)→(FN,C,FH,FW)
        dx=np.dot(dout,self.col_W.T)    #dot((N*out_h*out_w,FN),(FN,C*FH*FW))→(N*out_h*out_w,C*FH*FW)
        dx=col2im(dx,self.padded_x.shape,FH,FW,self.stride,0)
        
        dx=dx[self.mask].reshape(*self.original_shape)
        
        self.grads[0][...]=dW
        self.grads[1][...]=db
        
        return dx
        
        
class Repeat:
    def __init__(self,input_shape,param_list):
        '''
        param_list=[image_h,image_w]
        
        '''
        self.H,self.W=param_list
        C,=input_shape
        self.output_shape=(C,self.H,self.W)
        self.params,self.grads=[],[]
        
    def forward(self,x):
        N,C=x.shape
        x=x.reshape(N,C,1)
        x=np.repeat(x,self.H,axis=1)
        x=np.repeat(x,self.W,axis=2)
        x=x.reshape(N,C,self.H,-1)
        return x
        
    def backward(self,dout):
        N,C,H,W=dout.shape
        dout=dout.reshape(N,C,H*W)
        dout=dout.sum(axis=2)
        return dout
        
import sys,os
sys.path.append('..')
from common import config
from common.np import *
from common.layer_dictionary import *
from common.layers import *
from common.optimizer import *
import pickle
from common.util import to_cpu,to_gpu

class DeepConvResNet:
    def __init__(self,input_shape,
               res_layer_list_1,res_layer_list_2,res_layer_list_3,
               output_size):
        
        #Layerの生成
        self.layers=[]
        self.layers.append(ConvResNet(input_shape,res_layer_list_1))
        self.layers.append(Relu(self.layers[-1].output_shape,[]))
        self.layers.append(Pooling(self.layers[-1].output_shape,[2,2,0,2]))
        self.layers.append(ConvResNet(self.layers[-1].output_shape,res_layer_list_2))
        self.layers.append(Relu(self.layers[-1].output_shape,[]))
        self.layers.append(Pooling(self.layers[-1].output_shape,[2,2,0,2]))
        self.layers.append(ConvResNet(self.layers[-1].output_shape,res_layer_list_3))
        self.layers.append(Relu(self.layers[-1].output_shape,[]))
        self.layers.append(GAP(self.layers[-1].output_shape,[]))
        self.layers.append(Affine(self.layers[-1].output_shape,[output_size,2.]))

        self.loss_layer=SoftmaxWithLoss()
        
        self.params,self.grads=[],[]
        for layer in self.layers:
            self.params+=layer.params
            self.grads+=layer.grads

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, (BatchNormalization,ConvResNet)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def forward(self, x, t):
        y = self.predict(x, train_flg=True)
        loss=self.loss_layer.forward(y, t)
        return loss
        
    def backward(self,dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def save_params(self,file_name='DeepConvRes.pkl'):
        params = [p.astype(np.float16) for p in self.params]

        if GPU:
            params = [to_cpu(p) for p in params]

        with open(file_name,'wb') as f:
            pickle.dump(params,f)

    def load_params(self,file_name='DeepConvRes.pkl'):
        with open(file_name,'rb') as f:
            params=pickle.load(f)

        params = [p.astype('f') for p in params]
        if GPU:
            params = [to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
import sys,os
sys.path.append('..')
from common import config
from common.np import *
from common import layer_dictionary
from common.optimizer import *
import pickle
from common.util import to_cpu,to_gpu,show_distribution

class BaseModel:
    def __init__(self,input_shape,layer_list,loss_layer=None,show_distribution=False):
        '''
        layer_list=[['conv',param_list],['relu',param_list],['conv',param_list],['relu',param_list],['pool',param_list],
                    ['conv',param_list],['relu',param_list],['conv',param_list],['relu',param_list],['pool',param_list],
                    ['conv',param_list],['relu',param_list],['conv',param_list],['relu',param_list],['gap',param_list],
                    ['affine',param_list]]
        
        '''
        layer_dict=layer_dictionary.layer_dict
        loss_layer_dict=layer_dictionary.loss_layer_dict
        
        #Layerの生成
        self.layers=[]
        for i,layer in enumerate(layer_list):
            self.layers.append(layer_dict[layer[0]](input_shape,layer[1]))
            input_shape=self.layers[i].output_shape
            
        self.loss_layer=None
        if loss_layer is not None:
            self.loss_layer=loss_layer_dict[loss_layer]()
        
        self.params,self.grads=[],[]
        for layer in self.layers:
            self.params+=layer.params
            self.grads+=layer.grads
            
        self.show_distribution=show_distribution

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, (BatchNormalization,ConvResNet)):
                x = layer.forward(x, train_flg)
                isnan=np.isnan(x)
                if np.count_nonzero(isnan)!=0:
                    print(layer.__class__.__name__+':xにnanが含まれます')
                if self.show_distribution:
                    print(layer.__class__.__name__)
                    show_distribution(x)
            else:
                x = layer.forward(x)
                isnan=np.isnan(x)
                if np.count_nonzero(isnan)!=0:
                    print(layer.__class__.__name__+':xにnanが含まれます')
                if self.show_distribution:
                    print(layer.__class__.__name__)
                    show_distribution(x)

        return x

    def forward(self, x, t):
        y = self.predict(x, train_flg=True)
        if self.loss_layer is not None:
            loss=self.loss_layer.forward(y, t)
            isnan=np.isnan(loss)
            if isnan:
                print('lossがnanです')
            return loss
        else:
            return y
        
    def backward(self,dout):
        if self.loss_layer is not None:
            dout = self.loss_layer.backward(dout)
            isnan=np.isnan(dout)
            if np.count_nonzero(isnan)!=0:
                print(self.loss_layer.__class__.__name__+':勾配にnanが含まれます')
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            isnan=np.isnan(dout)
            if np.count_nonzero(isnan)!=0:
                print(layer.__class__.__name__+':勾配にnanが含まれます')
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

    def save_params(self,file_name='params.pkl'):
        params = [p.astype(np.float16) for p in self.params]

        if GPU:
            params = [to_cpu(p) for p in params]

        with open(file_name,'wb') as f:
            pickle.dump(params,f)

    def load_params(self,file_name='params.pkl'):
        with open(file_name,'rb') as f:
            params=pickle.load(f)

        params = [p.astype('f') for p in params]
        if GPU:
            params = [to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
from common import config
from common.np import *
from common.base_model import BaseModel
from common.layer_dictionary import *
from common.util import to_cpu,to_gpu
import matplotlib.pyplot as plt

class AutoEncoder(BaseModel):
    def __init__(self,input_shape,enc_layer_list,dec_layer_list,loss_layer,weight_decay=None,weight_decay_lambda=0,show_distribution=False):
        
        self.encoder=BaseModel(input_shape,enc_layer_list,loss_layer=None,show_distribution=show_distribution)
        self.decoder=BaseModel(self.encoder.layers[-1].output_shape,dec_layer_list,loss_layer=None,show_distribution=show_distribution)
        self.loss_layer=loss_layer_dict[loss_layer]()

        self.layers=[]
        self.layers=self.encoder.layers+self.decoder.layers
        
        self.params,self.grads=[],[]
        for layer in self.layers:
            self.params+=layer.params
            self.grads+=layer.grads
            
        self.weight_decay=weight_decay
        self.weight_decay_lambda=weight_decay_lambda
        self.show_distribution=show_distribution

        
    def generate(self,x,original_img_shape):
        train_flg=False
        y=self.decoder.forward(x,train_flg)
        y=y.reshape(*original_img_shape).transpose(1,2,0)
        return y
    
    
    def show_predict(self,x,original_img_shape,figsize=(10,20)):
        if GPU:
            x=to_gpu(x)
        x=np.array([x[np.random.randint(len(x))]])
        y=self.predict(x)
        
        x=x.reshape(*original_img_shape).transpose(1,2,0)
        y=y.reshape(*original_img_shape).transpose(1,2,0)
        
        if GPU:
            x=to_cpu(x)
            y=to_cpu(y)
        
        fig,ax=plt.subplots(1,2,figsize=figsize,facecolor='w')
        ax[0].imshow(x)
        ax[1].imshow(y)
        plt.show()
        
        
    def show_generate(self,start,stop,step,original_img_shape,figsize=(10,10)):
        xs=np.arange(start,stop,step)
        ys=np.arange(start,stop,step)
        
        if GPU:
            xs=to_cpu(xs)
            ys=to_cpu(ys)
        
        fig, ax = plt.subplots(len(xs), len(ys), figsize=figsize)
        fig.subplots_adjust(wspace=0,hspace=0)
        
        for j ,y in enumerate(ys):
            for i,x in enumerate(xs):
                a=np.array([[x,y]])
                z=self.generate(a,original_img_shape)
                z=z.reshape(*original_img_shape).transpose(1,2,0)
                if GPU:
                    z=to_cpu(z)
                ax[i][j].imshow(z)
                ax[i][j].axis('off')

        plt.show()
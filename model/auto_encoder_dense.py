from common.np import *
from common.layers_old import *
from common.util import to_cpu,to_gpu,show_distribution
import matplotlib.pyplot as plt
import pickle

class Encoder:
    def __init__(self,input_size,hidden_size_list,output_size,generate_activation,activation,use_batchnorm,show_distribution):
        generate_activation_layer = {'sigmoid': Sigmoid,'tanh':Tanh}
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu,'tanh':Tanh}
        if activation=='relu':
            weight_init_std=2.0
        else:
            weight_init_std=1.0

        rn=np.random.randn

        if len(hidden_size_list)!=0:
            affine_W=rn(input_size,hidden_size_list[0])*np.sqrt(weight_init_std/input_size).astype('f')
            affine_b=np.zeros(hidden_size_list[0]).astype('f')
            self.layers=[]
            self.layers.append(Affine(affine_W,affine_b))
            if use_batchnorm:
                gamma=np.ones(hidden_size_list[0]).astype('f')
                beta=np.zeros(hidden_size_list[0]).astype('f')
                self.layers.append(BatchNormalization(gamma=gamma,beta=beta))
            self.layers.append(activation_layer[activation]())
            
            for i in range(len(hidden_size_list)-1): 
                affine_W=rn(hidden_size_list[i],hidden_size_list[i+1])*np.sqrt(weight_init_std/hidden_size_list[i]).astype('f')
                affine_b=np.zeros(hidden_size_list[i+1]).astype('f')
                self.layers.append(Affine(affine_W,affine_b))
                if use_batchnorm:
                    gamma=np.ones(hidden_size_list[i+1]).astype('f')
                    beta=np.zeros(hidden_size_list[i+1]).astype('f')
                    self.layers.append(BatchNormalization(gamma=gamma,beta=beta))
                self.layers.append(activation_layer[activation]())
                
            affine_W=rn(hidden_size_list[-1],output_size)*np.sqrt(weight_init_std/hidden_size_list[-1]).astype('f')
            affine_b=np.zeros(output_size).astype('f')
            self.layers.append(Affine(affine_W,affine_b))
            if use_batchnorm:
                gamma=np.ones(output_size).astype('f')
                beta=np.zeros(output_size).astype('f')
                self.layers.append(BatchNormalization(gamma=gamma,beta=beta))
            self.layers.append(generate_activation_layer[generate_activation]())
            
        else:
            affine_W=rn(input_size,output_size)*np.sqrt(weight_init_std/input_size).astype('f')
            affine_b=np.zeros(output_size).astype('f')
            self.layers=[]
            self.layers.append(Affine(affine_W,affine_b))
            if use_batchnorm:
                gamma=np.ones(output_size).astype('f')
                beta=np.zeros(output_size).astype('f')
                self.layers.append(BatchNormalization(gamma=gamma,beta=beta))
            self.layers.append(generate_activation_layer[generate_activation]())

        self.params,self.grads=[],[]

        for layer in self.layers:
            self.params+=layer.params
            self.grads+=layer.grads
            
        self.show_distribution=show_distribution

    def forward(self,x,train_flg=True):
        for layer in self.layers:
            if isinstance(layer,BatchNormalization):
                x=layer.forward(x,train_flg)
                if self.show_distribution:
                    print(layer.__class__.__name__)
                    show_distribution(x)
            x=layer.forward(x)
            if self.show_distribution and (layer.__class__.__name__ in ['Sigmoid','Relu','Tanh']):
                print(layer.__class__.__name__)
                show_distribution(x)
        return x

    def backward(self,dout):
        for layer in reversed(self.layers):
            dout=layer.backward(dout)
            if self.show_distribution and isinstance(layer,Affine):
                print('dW')
                show_distribution(layer.grads[0])
                
        return dout
        
        
class Decoder:
    def __init__(self,input_size,hidden_size_list,output_size,activation,use_batchnorm,show_distribution):
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu,'tanh':Tanh}
        if activation=='relu':
            weight_init_std=2.0
        else:
            weight_init_std=1.0

        rn=np.random.randn
        if len(hidden_size_list)!=0:
            affine_W=rn(input_size,hidden_size_list[0])*np.sqrt(weight_init_std/input_size).astype('f')
            affine_b=np.zeros(hidden_size_list[0]).astype('f')
            self.layers=[]
            self.layers.append(Affine(affine_W,affine_b))
            if use_batchnorm:
                gamma=np.ones(hidden_size_list[0]).astype('f')
                beta=np.zeros(hidden_size_list[0]).astype('f')
                self.layers.append(BatchNormalization(gamma=gamma,beta=beta))
            self.layers.append(activation_layer[activation]())

            for i in range(len(hidden_size_list)-1): 
                affine_W=rn(hidden_size_list[i],hidden_size_list[i+1])*np.sqrt(weight_init_std/hidden_size_list[i]).astype('f')
                affine_b=np.zeros(hidden_size_list[i+1]).astype('f')
                self.layers.append(Affine(affine_W,affine_b))
                if use_batchnorm:
                    gamma=np.ones(hidden_size_list[i+1]).astype('f')
                    beta=np.zeros(hidden_size_list[i+1]).astype('f')
                    self.layers.append(BatchNormalization(gamma=gamma,beta=beta))
                self.layers.append(activation_layer[activation]())
                
            affine_W=rn(hidden_size_list[-1],output_size)*np.sqrt(weight_init_std/hidden_size_list[-1]).astype('f')
            affine_b=np.zeros(output_size).astype('f')
            self.layers.append(Affine(affine_W,affine_b))
            if use_batchnorm:
                gamma=np.ones(output_size).astype('f')
                beta=np.zeros(output_size).astype('f')
                self.layers.append(BatchNormalization(gamma=gamma,beta=beta))
            self.layers.append(activation_layer[activation]())
            
        else:
            affine_W=rn(input_size,output_size)*np.sqrt(weight_init_std/input_size).astype('f')
            affine_b=np.zeros(output_size).astype('f')
            self.layers=[]
            self.layers.append(Affine(affine_W,affine_b))
            if use_batchnorm:
                gamma=np.ones(output_size).astype('f')
                beta=np.zeros(output_size).astype('f')
                self.layers.append(BatchNormalization(gamma=gamma,beta=beta))
            self.layers.append(activation_layer[activation]())

        self.params,self.grads=[],[]

        for layer in self.layers:
            self.params+=layer.params
            self.grads+=layer.grads

        self.show_distribution=show_distribution

    def forward(self,x,train_flg=True):
        for layer in self.layers:
            if isinstance(layer,BatchNormalization):
                x=layer.forward(x,train_flg)
                if self.show_distribution:
                    print(layer.__class__.__name__)
                    show_distribution(x)
            x=layer.forward(x)
            if self.show_distribution and (layer.__class__.__name__ in ['Sigmoid','Relu','Tanh']):
                print(layer.__class__.__name__)
                show_distribution(x)
        return x

    def backward(self,dout):
        for layer in reversed(self.layers):
            dout=layer.backward(dout)
            if self.show_distribution and isinstance(layer,Affine):
                print('dW')
                show_distribution(layer.grads[0])
                # print('db')
                # show_distribution(layer.grads[1])
        return dout
        
        
class AE:
    def __init__(self,input_size=784,hidden_list_enc=[],generate_size=2,
                 hidden_list_dec=[],output_size=784,generate_activation='sigmoid',activation='relu',
                 use_batchnorm=False,show_distribution=False):
        self.encoder=Encoder(input_size,hidden_list_enc,generate_size,generate_activation,activation,use_batchnorm,show_distribution)
        self.decoder=Decoder(generate_size,hidden_list_dec,output_size,activation,use_batchnorm,show_distribution)
        self.loss_layer=MSE()

        self.layers=[]
        self.layers+=self.encoder.layers+self.decoder.layers

        self.params,self.grads=[],[]
        self.params+=self.encoder.params+self.decoder.params
        self.grads+=self.encoder.grads+self.decoder.grads

    def predict(self,x,train_flg=False):
        x=self.encoder.forward(x,train_flg)
        x=self.decoder.forward(x,train_flg)
        return x

    def forward(self,x,t,train_flg=True):
        score=self.predict(x,train_flg)
        loss=self.loss_layer.forward(score,t)
        return loss

    def backward(self,dout=1):
        dx=self.loss_layer.backward(dout)
        dx=self.decoder.backward(dx)
        dx=self.encoder.backward(dx)
        return dx

    def generate(self,x,original_img_shape=(28,28),train_flg=False):
        y=self.decoder.forward(x,train_flg)
        y=y.reshape(*original_img_shape)
        return y


    def show_predict(self,x,original_img_shape=(28,28),figsize=(10,20)):
        if GPU:
            x=to_gpu(x)
        x=np.array([x[np.random.randint(len(x))]])
        y=self.predict(x)
        
        x=x.reshape(*original_img_shape)
        y=y.reshape(*original_img_shape)
        
        if GPU:
            x=to_cpu(x)
            y=to_cpu(y)
        
        fig,ax=plt.subplots(1,2,figsize=figsize,facecolor='w')
        plt.gray()
        ax[0].imshow(x)
        ax[1].imshow(y)
        plt.show()
        
        
    def show_generate(self,start,stop,step,original_img_shape=(28,28),figsize=(10,10)):
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
                z=self.generate(a,original_img_shape,train_flg=False)
                z=z.reshape(*original_img_shape)
                if GPU:
                    z=to_cpu(z)
                ax[i][j].imshow(z,cmap='gray')
                ax[i][j].axis('off')
        fig.show()


    def save_params(self,file_name='AE.pkl'):
        params = [p.astype(np.float16) for p in self.params]

        if GPU:
            params = [to_cpu(p) for p in params]

        with open(file_name,'wb') as f:
            pickle.dump(params,f)

    def load_params(self,file_name='AE.pkl'):
        with open(file_name,'rb') as f:
            params=pickle.load(f)

        params = [p.astype('f') for p in params]
        if GPU:
            params = [to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
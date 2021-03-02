from common import config
from common.np import *
from common.base_model import BaseModel
from common.layer_dictionary import *
from common.util import to_cpu,to_gpu
import matplotlib.pyplot as plt

class Unet(BaseModel):
    def __init__(self,input_shape,conv1_in_list,conv2_in_list,conv3_in_list,conv4_in_list,
                 conv_bottom_list,conv4_out_list,conv3_out_list,conv2_out_list,conv1_out_list,
                 loss_layer,weight_decay=None,weight_decay_lambda=0,show_distribution=False):
        
        self.conv1_in=BaseModel(input_shape=input_shape,layer_list=conv1_in_list,loss_layer=None,weight_decay=weight_decay,
                                weight_decay_lambda=weight_decay_lambda,show_distribution=show_distribution)
        output_shape=self.conv1_in.layers[-1].output_shape
        self.pool1=layer_dict['pool'](output_shape,[2,2,0,2])
        output_shape=self.pool1.output_shape
        self.conv2_in=BaseModel(input_shape=output_shape,layer_list=conv2_in_list,loss_layer=None,weight_decay=weight_decay,
                                weight_decay_lambda=weight_decay_lambda,show_distribution=show_distribution)
        output_shape=self.conv2_in.layers[-1].output_shape
        self.pool2=layer_dict['pool'](output_shape,[2,2,0,2])
        output_shape=self.pool2.output_shape
        self.conv3_in=BaseModel(input_shape=output_shape,layer_list=conv3_in_list,loss_layer=None,weight_decay=weight_decay,
                                weight_decay_lambda=weight_decay_lambda,show_distribution=show_distribution)
        output_shape=self.conv3_in.layers[-1].output_shape
        self.pool3=layer_dict['pool'](output_shape,[2,2,0,2])
        output_shape=self.pool3.output_shape
        self.conv4_in=BaseModel(input_shape=output_shape,layer_list=conv4_in_list,loss_layer=None,weight_decay=weight_decay,
                                weight_decay_lambda=weight_decay_lambda,show_distribution=show_distribution)
        output_shape=self.conv4_in.layers[-1].output_shape
        self.pool4=layer_dict['pool'](output_shape,[2,2,0,2])
        output_shape=self.pool4.output_shape
        self.conv_bottom=BaseModel(input_shape=output_shape,layer_list=conv_bottom_list,loss_layer=None,weight_decay=weight_decay,
                                weight_decay_lambda=weight_decay_lambda,show_distribution=show_distribution)
        output_shape=self.conv_bottom.layers[-1].output_shape
        C,H,W=output_shape
        output_shape=(C+self.conv4_in.layers[-1].output_shape[0],H,W)
        self.conv4_out=BaseModel(input_shape=output_shape,layer_list=conv4_out_list,loss_layer=None,weight_decay=weight_decay,
                                weight_decay_lambda=weight_decay_lambda,show_distribution=show_distribution)
        output_shape=self.conv4_out.layers[-1].output_shape
        C,H,W=output_shape
        output_shape=(C+self.conv3_in.layers[-1].output_shape[0],H,W)
        self.conv3_out=BaseModel(input_shape=output_shape,layer_list=conv3_out_list,loss_layer=None,weight_decay=weight_decay,
                                weight_decay_lambda=weight_decay_lambda,show_distribution=show_distribution)
        output_shape=self.conv3_out.layers[-1].output_shape
        C,H,W=output_shape
        output_shape=(C+self.conv2_in.layers[-1].output_shape[0],H,W)
        self.conv2_out=BaseModel(input_shape=output_shape,layer_list=conv2_out_list,loss_layer=None,weight_decay=weight_decay,
                                weight_decay_lambda=weight_decay_lambda,show_distribution=show_distribution)
        output_shape=self.conv2_out.layers[-1].output_shape
        C,H,W=output_shape
        output_shape=(C+self.conv1_in.layers[-1].output_shape[0],H,W)
        self.conv1_out=BaseModel(input_shape=output_shape,layer_list=conv1_out_list,loss_layer=None,weight_decay=weight_decay,
                                weight_decay_lambda=weight_decay_lambda,show_distribution=show_distribution)

        self.loss_layer=loss_layer_dict[loss_layer]()

        self.layers=[]
        self.layers=self.conv1_in.layers+[self.pool1]\
                    +self.conv2_in.layers+[self.pool2]\
                    +self.conv3_in.layers+[self.pool3]\
                    +self.conv4_in.layers+[self.pool4]\
                    +self.conv_bottom.layers\
                    +self.conv4_out.layers\
                    +self.conv3_out.layers\
                    +self.conv2_out.layers\
                    +self.conv1_out.layers

        self.params,self.grads=[],[]
        for layer in self.layers:
            self.params+=layer.params
            self.grads+=layer.grads

        self.conv1x_shape=None
        self.conv2x_shape=None
        self.conv3x_shape=None
        self.conv4x_shape=None


    def predict(self,x,train_flg=False):
        x=self.conv1_in.predict(x,train_flg)
        conv1_x=x
        self.conv1x_shape=conv1_x.shape
        x=self.pool1.forward(x)
        x=self.conv2_in.predict(x,train_flg)
        conv2_x=x
        self.conv2x_shape=conv2_x.shape
        x=self.pool2.forward(x)
        x=self.conv3_in.predict(x,train_flg)
        conv3_x=x
        self.conv3x_shape=conv3_x.shape
        x=self.pool3.forward(x)
        x=self.conv4_in.predict(x,train_flg)
        conv4_x=x
        self.conv4x_shape=conv4_x.shape
        x=self.pool4.forward(x)
        x=self.conv_bottom.predict(x,train_flg)

        conv4_x=self.crop_image(conv4_x,x)
        x=np.concatenate([x,conv4_x],axis=1)
        x=self.conv4_out.predict(x,train_flg)

        conv3_x=self.crop_image(conv3_x,x)
        x=np.concatenate([x,conv3_x],axis=1)
        x=self.conv3_out.predict(x,train_flg)

        conv2_x=self.crop_image(conv2_x,x)
        x=np.concatenate([x,conv2_x],axis=1)
        x=self.conv2_out.predict(x,train_flg)

        conv1_x=self.crop_image(conv1_x,x)
        x=np.concatenate([x,conv1_x],axis=1)
        x=self.conv1_out.predict(x,train_flg)

        return x


    def forward(self,x,t):
        x=self.predict(x=x,train_flg=True)
        t=self.crop_image(t,x)
        loss=self.loss_layer.forward(x=x,t=t)

        return loss


    def backward(self,dout=1):
        dout=self.loss_layer.backward(dout)
        dout=self.conv1_out.backward(dout)

        # dx1=dout[:,:self.conv1x_shape[1],:,:]
        # conv1_dx=dout[:,self.conv1x_shape[1]:,:,:]

        dx1,conv1_dx=self.split_pad(dout,self.conv1x_shape)
        dout=self.conv2_out.backward(dx1)

        # dx2=dout[:,:self.conv2x_shape[1],:,:]
        # conv2_dx=dout[:,self.conv2x_shape[1]:,:,:]

        dx2,conv2_dx=self.split_pad(dout,self.conv2x_shape)
        dout=self.conv3_out.backward(dx2)

        # dx3=dout[:,:self.conv3x_shape[1],:,:]
        # conv3_dx=dout[:,self.conv3x_shape[1]:,:,:]

        dx3,conv3_dx=self.split_pad(dout,self.conv3x_shape)
        dout=self.conv4_out.backward(dx3)

        # dx4=dout[:,:self.conv4x_shape[1],:,:]
        # conv4_dx=dout[:,self.conv4x_shape[1]:,:,:]

        dx4,conv4_dx=self.split_pad(dout,self.conv4x_shape)
        dout=self.conv_bottom.backward(dx4)

        dout=self.pool4.backward(dout)
        dout+=conv4_dx
        dout=self.conv4_in.backward(dout)
        dout=self.pool3.backward(dout)
        dout+=conv3_dx
        dout=self.conv3_in.backward(dout)
        dout=self.pool2.backward(dout)
        dout+=conv2_dx
        dout=self.conv2_in.backward(dout)
        dout=self.pool1.backward(dout)
        dout+=conv1_dx
        dout=self.conv1_in.backward(dout)

        return dout

    
    def crop_image(self,x,x_out):
        N,C,H,W=x.shape
        N,out_c,out_h,out_w=x_out.shape
        q_h,mod_h=divmod(H-out_h,2)
        q_w,mod_w=divmod(W-out_w,2)
        x=x[:,:,q_h:-(q_h+mod_h),q_w:-(q_w+mod_w)]
        return x


    def split_pad(self,dout,x_shape):
        N,C,H,W=x_shape
        N,out_c,out_h,out_w=dout.shape
        #split
        dx1=dout[:,:out_c-C,:,:]
        dx2=dout[:,out_c-C:,:,:]
        #pad
        q_h,mod_h=divmod(H-out_h,2)
        q_w,mod_w=divmod(W-out_w,2)
        dx2=np.pad(dx2,[(0,0),(0,0),(q_h,q_h+mod_h),(q_w,q_w+mod_w)])
        return dx1,dx2
        










        
        





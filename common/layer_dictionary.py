# coding: utf-8
from common.layers import *
from common.loss_layer import *

layer_dict={
            'relu':Relu,'sigmoid':Sigmoid, 'tanh':Tanh,
            'conv':Convolution,'deconv':Deconvolution,'pool':Pooling,'affine':Affine,
            'batchnorm':BatchNormalization,'gap':GAP,
            'convres':ConvResNet,'repeat':Repeat,'dropout':Dropout
        }
        
loss_layer_dict={
            'mse':MSE,'softmax':SoftmaxWithLoss
        }
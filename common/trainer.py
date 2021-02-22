# coding: utf-8
import sys
sys.path.append('..')
from common import config
import numpy
import time
import matplotlib.pyplot as plt
from common.np import *
from common.util import clip_grads,to_cpu,to_gpu
import pickle


class Trainer:
    def __init__(self, model, optimizer,file_name=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0
        self.train_acc_list = []
        self.test_acc_list = []
        self.file_name=file_name
        
    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20,eval_accuracy=False,eval_sample_num=None,x_test=None,t_test=None):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # シャッフル
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                dout=1
                model.backward(dout)
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    if eval_accuracy:
                        x_sample=x
                        t_sample=t
                        x_test_sample=x_test
                        t_test_sample=t_test
                        if eval_sample_num is not None:
                            x_sample=x_sample[:eval_sample_num]
                            t_sample=t_sample[:eval_sample_num]
                            x_test_sample=x_test_sample[:eval_sample_num]
                            t_test_sample=t_test_sample[:eval_sample_num]

                        train_acc=self.model.accuracy(x_sample,t_sample)
                        test_acc=self.model.accuracy(x_test_sample,t_test_sample)
                        self.train_acc_list.append(train_acc)
                        self.test_acc_list.append(test_acc)
                        print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f | train_acc %.2f | test_acc %.2f'
                              % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss,train_acc,test_acc))
                    else:
                        print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
                              % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0
            if self.file_name is not None:
                file_name_loss=self.file_name+'_'+str(self.loss_list[-1])+'.pkl'
                model.save_params(file_name_loss)
                self.save_loss()
                if eval_accuracy:
                    self.save_acc()
                
            self.current_epoch += 1
        
        if eval_accuracy:
            train_acc=self.model.accuracy(x,t)
            test_acc=self.model.accuracy(x_test,t_test)
            print("="*50)
            print("Final train acc: %.2f | Final test acc: %.2f"%(train_acc,test_acc))


    def plot(self, ylim=None):
        if GPU:
            loss_list=to_cpu(self.loss_list)
        
        x = numpy.arange(len(loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()
        
    def save_loss(self):
        train_loss_list=self.loss_list.copy()
        if GPU:
            train_loss_list=[to_cpu(p) for p in train_loss_list]

        loss_list=[train_loss_list]
            
        with open(self.file_name+'_loss.pkl','wb') as f:
            pickle.dump(loss_list,f)
            
    def save_acc(self):
        train_acc_list=self.train_acc_list.copy()
        test_acc_list=self.test_acc_list.copy()
        if GPU:
            train_acc_list=[to_cpu(p) for p in train_acc_list]
            test_acc_list=[to_cpu(p) for p in test_acc_list]
            
        acc_list=[train_acc_list,test_acc_list]
        with open(self.file_name+'_acc.pkl','wb') as f:
            pickle.dump(acc_list,f) 


def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads
# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000 #迭代次数
train_size = x_train.shape[0] #训练样本数
batch_size = 100 #批次数
learning_rate = 0.1 #学习速率

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1) #10次

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) #随机选出batch_size个数
    x_batch = x_train[batch_mask]  #批次输入
    t_batch = t_train[batch_mask]  #批次输出
    
    # 梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) #梯度更新
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'): #梯度下降
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch) #损失
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

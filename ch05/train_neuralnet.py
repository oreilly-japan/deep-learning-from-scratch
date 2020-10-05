# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import Network
from layers import LayerDifinition

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = Network([
    LayerDifinition('affine', {'input': 784, 'output': 100, 'weight_init_std': 0.01}),
    LayerDifinition('relu', {}),
    LayerDifinition('affine', {'input': 100, 'output': 20, 'weight_init_std': 0.01}),
    LayerDifinition('relu', {}),
    LayerDifinition('affine', {'input': 20, 'output': 10, 'weight_init_std': 0.01}),
    LayerDifinition('softmax_with_loss', {}),
])

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    network.learning(x_batch, t_batch, learning_rate)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

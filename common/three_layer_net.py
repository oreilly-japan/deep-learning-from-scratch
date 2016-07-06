# coding: utf-8
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from common.functions import *
from common.layers import *
from common.gradient import numerical_gradient

class ThreeLayerNet():

    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std = 0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size_list[0])
        self.params['B1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size_list[0], hidden_size_list[1]) 
        self.params['B2'] = np.zeros(output_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size_list[1], output_size) 
        self.params['B3'] = np.zeros(output_size)

        self.layers = []
        self.layers.append(AffineLayer(self.params['W1'], self.params['B1']))
        self.layers.append(SigmoidLayer())
        self.layers.append(AffineLayer(self.params['W2'], self.params['B2']))
        self.layers.append(SigmoidLayer())
        self.layers.append(AffineLayer(self.params['W3'], self.params['B3']))
        self.layers.append(SigmoidLayer())

        
    def forward(self, X):
        W1, W2 = self.params['W1'], self.params['W2']
        B1, B2 = self.params['B1'], self.params['B2']
    
        A1 = np.dot(X, W1) + B1
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, W2) + B2
        Y = softmax(A2)
        
        return Y
        
    # X:入力データ, T:教師データ
    def loss(self, X, T):
        Y = self.forward(X)
        
        return cross_entropy_error(Y, T)
    
    def accuracy(self, X, T):
        Y = self.forward(X)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1 : T = np.argmax(T, axis=1)
        
        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy
        
    # X:入力データ, T:教師データ
    def numerical_gradient(self, X, T):
        loss_W = lambda W: self.loss(X, T)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['B1'] = numerical_gradient(loss_W, self.params['B1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['B2'] = numerical_gradient(loss_W, self.params['B2'])
        
        return grads
        
    def gradient(self, X, T):
        W1, W2 = self.params['W1'], self.params['W2']
        B1, B2 = self.params['B1'], self.params['B2']
        grads = {}
        
        batch_num, _ = X.shape
        
        # forward
        A1 = np.dot(X, W1) + B1
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, W2) + B2
        Y = softmax(A2)
        
        # backward
        dY = (Y - T) / batch_num
        grads['W2'] = np.dot(Z1.T, dY)
        grads['B2'] = np.sum(dY, axis=0)
        
        dA1 = np.dot(dY, W2.T)
        dZ1 = sigmoid_grad(A1) * dA1
        grads['W1'] = np.dot(X.T, dZ1)
        grads['B1'] = np.sum(dZ1, axis=0)
        
        return grads
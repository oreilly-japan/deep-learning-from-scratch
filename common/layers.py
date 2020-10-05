# coding: utf-8
import sys
import numpy as np
from common.functions import *
from common.util import im2col, col2im

# factory
class LayerDifinition:
    def __init__(self, name, options):
        self.name = name
        self.options = options

    def create(self):
        if self.name == Relu.name:
            return Relu()
        elif self.name == Sigmoid.name:
            return Sigmoid()
        elif self.name == Affine.name:
            return Affine(self.options['input'], self.options['output'], self.options['weight_init_std'])
        elif self.name == SoftmaxWithLoss.name:
            return SoftmaxWithLoss()
        else:
            print("unknown layer")
            sys.exit()

class Relu:
    name = 'relu'

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

    def learning(self, learning_rate):
        pass


class Sigmoid:
    name = 'sigmoid'

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

    def learning(self, learning_rate):
        pass

class Affine:
    name = 'affine'

    def __init__(self, input, output, weight_init_std):
        self.W = weight_init_std * np.random.randn(input, output)
        self.b = np.zeros(output)

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx

    def learning(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

class SoftmaxWithLoss:
    name = 'softmax_with_loss'

    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

    def learning(self, learning_rate):
        pass

# coding: utf-8
import sys, os
import numpy as np
from functools import reduce


class Network:

    def __init__(self, definitions):
        self.all_layers = list(map(lambda x: x.create(), definitions))
        self.layers = self.all_layers[:-1]
        self.lastLayer = self.all_layers[-1]

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def learning(self, x, t, rate):
        # forward
        self.loss(x, t)

        # backward
        reduce(lambda acc, layer: layer.backward(acc), reversed(self.all_layers), 1)

        for l in self.layers:
            l.learning(rate)

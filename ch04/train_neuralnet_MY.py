import sys, os

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

import numpy as np
from common.functions import softmax
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
softmax(0.5)

# ハイパーパラメータ
iters_num: int = 10000
train_size: int = x_train.shape[0]
batch_size: int = 100
learning_rate: float = 0.1

# ニューラルネットワークの初期化
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1エポック当たりの繰り返し数
iter_per_epoch: float = max(train_size / batch_size, 1)


for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)  # 高速版

    # パラメータの更新
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1エポックごとに認識制度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


def draw_graph():
    markers = {"train": "o", "test": "s"}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label="train acc")
    plt.plot(x, test_acc_list, label="test acc", linestyle="--")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    draw_graph()

# ゼロから作る Deep Learning

---

![表紙](https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/images/deep-learning-from-scratch.png)

---

本リポジトリはオライリー・ジャパン発行書籍『[ゼロから作る Deep Learning](http://www.oreilly.co.jp/books/9784873117584/)』のサポートサイトです。

## ファイル構成

|フォルダ名 |説明                         |
|:--        |:--                          |
|ch01       |1章で使用するソースコード    |
|ch02       |2章で使用するソースコード    |
|...        |...                          |
|ch08       |8章で使用するソースコード    |
|common     |共通で使用するソースコード   |
|dataset    |データセット用のソースコード |


ソースコードの解説は本書籍をご覧ください。

## 必要条件
ソースコードを実行するには、下記のソフトウェアがインストールされている必要があります。

* Python 3.x
* NumPy
* Matplotlib

※Pythonのバージョンは、3系を利用します。

## 実行方法

各章のフォルダへ移動して、Pythonコマンドを実行します。

```
$ cd ch01
$ python man.py

$ cd ../ch05
$ python train_nueralnet.py
```

## ライセンス

本リポジトリのソースコードは[MITライセンス](http://www.opensource.org/licenses/MIT)です。
商用・非商用問わず、自由にご利用ください。

## 正誤表

下記の誤りがありました。お詫びして訂正いたします。

本ページに掲載されていない誤植など間違いを見つけた方は、[japan＠oreilly.co.jp](<mailto:japan＠oreilly.co.jp>)までお知らせください。

### 第4刷まで

|頁                |誤     |正     |
|:--               |:--    |:--    |
|目次、5章、索引   |連鎖率 |連鎖律 |
|4章 P.111 L.23    |続いて、勾配を求めてみましょう。これまでどおり、`numerical_gradient(f, x)`を使って勾配を求めます。 |続いて、勾配を求めてみましょう。これまでどおり、`numerical_gradient(f, x)`を使って勾配を求めます（ここで定義した`f(W)`という関数の引数`W`は、ダミーとして設けたものです。これは、`numerical_gradient(f, x)`が内部で`f(x)`を実行するため、それと整合性がとれるように`f(W)`を定義しました）。 |
|7章 P.236 L.10,14 |1989年 |1998年 |

### 第3刷まで

|頁           |誤  |正  |
|:--          |:-- |:-- |
|3章 P.76 L.4   |`ch03/nueralnet_mnist.py` |`ch03/neuralnet_mnist.py` |
|4章 P.89 L.19  |`>>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0]`<br>`>>> mean_squared_error(np.array(y), np.array(t))`<br>`0.72250000000000003` |`>>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]`<br>`>>> mean_squared_error(np.array(y), np.array(t))`<br>`0.59750000000000003` |
|4章 P.91 L.15  |`>>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0]`<br>`>>> cross_entropy_error(np.array(y), np.array(t))`<br>`2.3025850919940458` |`>>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]`<br>`>>> cross_entropy_error(np.array(y), np.array(t))`<br>`2.3025840929945458` |
|4章 P.117 L.29 |`ch04/train_nueralnet.py` |`ch04/train_neuralnet.py` |
|4章 P.118 L.7  |`# ハイパーパラメータ`<br>`iters_num = 10000`<br>`batch_size = 100`<br>`learning_rate = 0.1` |`# ハイパーパラメータ`<br>`iters_num = 10000`<br>`train_size = x_train.shape[0]`<br>`batch_size = 100`<br>`learning_rate = 0.1` |
|5章 P.127 L.23 |各ノードでは単純な計算をに集中することで、 |各ノードでは単純な計算に集中することで、 |
|5章 P.162 L.19 |`ch05/train_nueralnet.py` |`ch05/train_neuralnet.py` |

### 第2刷まで

|頁           |誤  |正  |
|:--          |:-- |:-- |
|2章 P.24 L.1 |両方が1のときだけ0を出力し、それ以外は0を出力します。 |両方が1のときだけ0を出力し、それ以外は1を出力します。 |

### 第1刷

|頁               |誤  |正  |
|:--              |:-- |:-- |
|7章 P.227 図7-20 |<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/images/fig07_20-old.png" alt="誤"> | <img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/images/fig07_20-new.png" alt="正"> |

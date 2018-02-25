# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # どこでも実行できるようにする

img = imread('../dataset/lena.png') # 画像の読み込み
plt.imshow(img)

plt.show()
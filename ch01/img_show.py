# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../dataset/lena.png') # 이미지 읽어오기
plt.imshow(img)

plt.show()

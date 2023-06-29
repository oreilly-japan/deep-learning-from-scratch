import numpy as np
import matplotlib.pyplot as plt

# データの生成
x = np.arange(0, 6, 0.1) # 0から6まで0.1刻みで生成
y1 = np.sin(x)
y2 = np.cos(x)

# グラフの描写
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label = "cos") # 破線で描画
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
# plt.legend()
plt.show()

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0              #first evaluate x > 0, after y is assigned bool 
    y = y.astype(np.int)
    return y

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)        #range of y-axis 
plt.show()


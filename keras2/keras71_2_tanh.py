import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

x = np.arange(-5, 5, 0.1)
y = tanh(x)

# print(x,"/",y)

plt.plot(x,y)
plt.grid()
plt.show()
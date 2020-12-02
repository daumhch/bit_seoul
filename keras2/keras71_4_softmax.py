import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


x = np.arange(1, 5)
y = softmax(x)

# print(x,"/",y)

ratio = y
labels = y

plt.pie(ratio, labels = labels, shadow=True)
plt.show()

#### softmax는 가장 마지막에 결과값을 분류하기 좋은 activation function이다


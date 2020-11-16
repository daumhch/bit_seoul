import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,) (10000,)

# print(x_train[0]) # 28x28 리스트 데이터
# print(y_train[0]) # 5

plt.imshow(x_train[1989],'gray')
plt.show()
# 0 = 5
# 30000 = 3
# 19999 = 2
# 1999 = 0
# 1989 = 1





import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train[0])
print("y_train[0]:", y_train[0])
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)

plt.imshow(x_train[0])
plt.show()


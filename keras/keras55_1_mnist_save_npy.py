import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

# 1.데이터
# 1.1 load_data
# 1.2 train_test_split
# 1.3 scaler
# 1.4 reshape
# 2.모델
# 3.컴파일 훈련
# 4.평가 예측


import numpy as np

# 1.1 load_data
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("origin x shape:",x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print("origin y shape:",y_train.shape, y_test.shape) # (60000,) (10000,)


# npy 저장하기
np.save('./data/mnist_x_train.npy', arr=x_train)
np.save('./data/mnist_x_test.npy', arr=x_test)
np.save('./data/mnist_y_train.npy', arr=y_train)
np.save('./data/mnist_y_test.npy', arr=y_test)


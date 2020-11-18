import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train[0])
# print("y_train[0]:", y_train[0])
print("x_train.shape:", x_train.shape) # x_train.shape: (50000, 32, 32, 3)
print("x_test.shape:", x_test.shape) # x_test.shape: (10000, 32, 32, 3) 
print("y_test.shape:", y_test.shape) # y_test.shape: (10000, 1)




# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)
print(y_train[0]) # [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]  


# train_test_split
# scaler
# reshape

# train_test_split
# 입력 데이터가 이미 train과 test로 나뉘어 있어서 별도로 나누지 않는다
# validation은 fit에서 vadidation_split으로 나눈다


# Scaler
# 선택은 아무거나, 최적이라 생각하는 주관적 판단
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
# print(x_train[0])



# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_train.shape[3])
print("reshape x:", x_train.shape, x_test.shape)




'''
# 2.모델
from tensorflow.keras.models import load_model
model = load_model('./save/weight_test02.h5')
model.summary()
'''

# 3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model = load_model('./model/cifar10-05-0.9968.hdf5')
# loss:  1.0141860246658325
# accuracy:  0.6521999835968018





# model = load_model('./save/model_test02_2.h5')
# loss:  2.945673942565918
# accuracy:  0.6484000086784363



# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("accuracy: ", accuracy)


y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
# print("y_predict:\n", y)
# print("y_test:\n", y_test)
# print("y_predict:\n", y_predict)



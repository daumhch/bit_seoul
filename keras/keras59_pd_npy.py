# 실습
# iris_ys2.csv 파일을 넘파이로 불러오기
# 불러온 데이터를 판다스로 저장(csv파일)하시오
# 파일명은 iris_ys2_pd.csv
# 모델 완성

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np
import pandas as pd

datasets_npy = np.loadtxt('./data/csv/iris_ys2.csv',delimiter=',')
print(type(datasets_npy)) # <class 'numpy.ndarray'>
print(datasets_npy.shape) # (150, 5)


datasets_pd = pd.DataFrame(datasets_npy)
print(type(datasets_pd)) # <class 'pandas.core.frame.DataFrame'>
print(datasets_pd.shape) # (150, 5)
print(datasets_pd.head())
#      0    1    2    3    4
# 0  5.1  3.5  1.4  0.2  0.0
# 1  4.9  3.0  1.4  0.2  0.0
# 2  4.7  3.2  1.3  0.2  0.0
# 3  4.6  3.1  1.5  0.2  0.0
# 4  5.0  3.6  1.4  0.2  0.0

datasets_pd.to_csv('./data/csv/iris_ys2_pd.csv')



# npy를 쪼개기
x = datasets_npy[:,:4]
y = datasets_npy[:,4:]

# print(x)
# print(y)

'''

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) # (150, 3)
print(y[0]) # [1. 0. 0.]


# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, random_state = 44)

print("after x_train.shape",x_train.shape) # (135, 4)
print("after x_test.shape",x_test.shape) # (15, 4)


# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) # fit하고
x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자
x_test = scaler.transform(x_test) # 사용할 수 있게 바꿔서 저장하자


# 1.4 reshape
# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test = x_test.reshape(x_test.shape[0],x_train.shape[1],1,1)
print("reshape x:", x_train.shape, x_test.shape) # (135, 4, 1, 1) (15, 4, 1, 1)





from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score



modelpath = './save/keras53_7_240_0.1012.hdf5'
model_save_path = "./save/keras53_7_iris_model.h5"
weights_save_path = './save/keras53_7_iris_weights.h5'

# 2.모델1=======================================================
from tensorflow.keras.models import load_model
model1 = load_model(model_save_path)

result1 = model1.evaluate(x_test, y_test, batch_size=128)

y_predict1 = model1.predict(x_test)
y_predict1 = np.argmax(y_predict1, axis=1)
y_recovery = np.argmax(y_test, axis=1)
print("RMSE_1:", RMSE(y_recovery, y_predict1))

r2_1 = r2_score(y_recovery, y_predict1)
print("R2_1:", r2_1)
# 2.모델1 끝=======================================================





# 2.모델2=======================================================
from tensorflow.keras.models import load_model
model2 = load_model(modelpath)

result2 = model2.evaluate(x_test, y_test, batch_size=128)

y_predict2 = model2.predict(x_test)
y_predict2 = np.argmax(y_predict2, axis=1)
y_recovery = np.argmax(y_test, axis=1)
print("RMSE_2:", RMSE(y_recovery, y_predict2))

r2_2 = r2_score(y_recovery, y_predict2)
print("R2_2:", r2_2)
# 2.모델2 끝=======================================================





# 2.모델3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
model3 = Sequential()
model3.add(Conv2D(16, (x_train.shape[1]), padding='same', input_shape=(x_train.shape[1],1,1) ))
model3.add(Flatten())
model3.add(Dense(64, activation = 'relu'))
model3.add(Dense(64, activation = 'relu'))
model3.add(Dense(3, activation='softmax'))
model3.summary()

# 3. 컴파일, 훈련
model3.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model3.load_weights(weights_save_path)

result3 = model3.evaluate(x_test, y_test, batch_size=128)

y_predict = model3.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_recovery = np.argmax(y_test, axis=1)
print("RMSE_3:", RMSE(y_recovery, y_predict))

r2_3 = r2_score(y_recovery, y_predict)
print("R2_3:", r2_3)

# 2.모델3 끝=======================================================

print("result1:", result1)
print("result2:", result2)
print("result3:", result3)
# result1: [0.03974771872162819, 1.0]
# result2: [0.052695903927087784, 0.9333333373069763]
# result3: [0.03974771872162819, 1.0]



'''





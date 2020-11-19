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
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print("x.shape:",x.shape) # (569, 30)
print("y.shape:",y.shape) # (569,)


# OneHotEncoding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) # (569, 2)
print(y[0]) # [1. 0.]


# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1)

print("after x_train.shape",x_train.shape) # (512, 30)
print("after x_test.shape",x_test.shape) # (57, 30)


# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) # fit하고
x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자
x_test = scaler.transform(x_test) # 사용할 수 있게 바꿔서 저장하자


# 1.4 reshape
# x가 2차원이라, 별도의 reshape 없이 그대로 x를 사용한다





from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score



modelpath = './save/keras53_8_76_0.0270.hdf5'
model_save_path = "./save/keras53_8_cancer_model.h5"
weights_save_path = './save/keras53_8_cancer_weights.h5'

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
model3.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],) ))
model3.add(Dense(512, activation='relu'))
model3.add(Dense(512, activation='relu'))
model3.add(Dense(512, activation='relu'))
model3.add(Dense(256, activation='relu'))
model3.add(Dense(256, activation='relu'))
model3.add(Dense(128, activation='relu'))
model3.add(Dense(128, activation='relu'))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(2, activation='sigmoid') )
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
# result1: [1.9002008002644288e-06, 0.0001822795020416379]
# result2: [8.003606490092352e-05, 0.0016799158183857799]
# result3: [0.0001973948092199862, 1.0]


print("RMSE_1:", RMSE(y_recovery, y_predict1))
print("R2_1:", r2_1)
print("RMSE_2:", RMSE(y_recovery, y_predict2))
print("R2_2:", r2_2)
print("RMSE_3:", RMSE(y_recovery, y_predict))
print("R2_3:", r2_3)

# RMSE_1: 0.0
# R2_1: 1.0
# RMSE_2: 0.0
# R2_2: 1.0
# RMSE_3: 0.0
# R2_3: 1.0
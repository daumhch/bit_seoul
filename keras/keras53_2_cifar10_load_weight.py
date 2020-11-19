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
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("origin x shape:",x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
print("origin y shape:",y_train.shape, y_test.shape) # (50000, 1) (10000, 1)       


# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)
print(y_train[0]) # [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]


# 1.2 train_test_split
# load_data할 때 이미 train과 test가 나뉘어 있으니 별도로 나누지 않는다
# validation은 fit에서 validation_split으로 적용한다


# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train) # fit하고
# x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자

# Scaler는 2차원 이하만 된다, 수동으로 바꾸자 (게다가 최소/최대값을 알고 있으니...)
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


# 1.4 reshape
# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_train.shape[3])
print("reshape x:", x_train.shape, x_test.shape)






from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score



modelpath = './save/keras53_2_39_0.5823.hdf5'
model_save_path = "./save/keras53_2_cifar10_model.h5"
weights_save_path = './save/keras53_2_cifar10_weights.h5'

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
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
model3 = Sequential()
model3.add( Conv2D(32, (3,3), padding='same', input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])) )
model3.add(Dropout(0.2))

model3.add( Conv2D(32, (3,3), padding='same', activation='relu') )
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Dropout(0.2))

model3.add( Conv2D(64, (3,3), padding='same', activation='relu') )
model3.add( Conv2D(64, (3,3), padding='same', activation='relu') )
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Dropout(0.3))

model3.add( Conv2D(128, (3,3), padding='same', activation='relu') )
model3.add( Conv2D(128, (3,3), padding='same', activation='relu') )
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Dropout(0.4))

model3.add(Flatten())
model3.add(Dense(512, activation = 'relu'))
model3.add(Dropout(0.2))
model3.add(Dense(10, activation = 'softmax') )
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

# result1: [0.7250201106071472, 0.8130999803543091]
# result2: [0.6283094882965088, 0.8064000010490417]
# result3: [0.7250201106071472, 0.8130999803543091]



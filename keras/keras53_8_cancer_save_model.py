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



modelpath = './model/keras53-8-{epoch:02d}-{val_loss:.4f}.hdf5'
model_save_path = "./save/keras53_8_cancer_model.h5"
weights_save_path = './save/keras53_8_cancer_weights.h5'

# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Dense(int(x_train.shape[1]/1), activation='relu', input_shape=(x_train.shape[1],) ))
model.add(Dense(int(x_train.shape[1]/2), activation='relu'))
model.add(Dense(int(x_train.shape[1]/3), activation='relu'))
model.add(Dense(int(x_train.shape[1]/4), activation='relu'))
model.add(Dense(int(x_train.shape[1]/5), activation='relu'))
model.add(Dense(int(x_train.shape[1]/6), activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.summary()



# 3. 컴파일, 훈련
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    mode='auto',
    verbose=2)

from tensorflow.keras.callbacks import ModelCheckpoint # 모델 체크 포인트
model_check_point = ModelCheckpoint(
    filepath=modelpath,
    monitor='val_loss',
    save_best_only=True,
    mode='auto')

model.fit(x_train, y_train,
    epochs=1000,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping, model_check_point])

model.save(model_save_path)
model.save_weights(weights_save_path)


# 4. 평가, 예측
result3 = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", result3[0])
print("accuracy: ", result3[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# print("y_predict:", y_predict)



y_recovery = np.argmax(y_test, axis=1)

# 사용자정의 RMSE 함수
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_recovery, y_predict))


# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y_recovery, y_predict)
print("R2:", r2)


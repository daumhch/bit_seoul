import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np

# 1.데이터
# 1.1 load_data
a = np.array(range(1,100))
size = 5

# split_x 함수
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) -size +1 ):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

dataset = split_x(a, size)

# Conv1D로 모델을 구성하시오
x = dataset[:,:4]
y = dataset[:,4:]
print("x.shape:",x.shape)
print("y.shape:",y.shape)


# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, random_state = 44)

print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)


# 1.3 scaler
# from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# 1.4 reshape
# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
print("reshape x:", x_train.shape, x_test.shape)


modelpath = './model/keras60_{epoch:02d}_{val_loss:.4f}.hdf5'
model_save_path = "./save/keras60_Conv1D_model.h5"
weights_save_path = './save/keras60_Conv1D_weights.h5'

# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D

model = Sequential()
model.add(Conv1D(75, 3, input_shape=(x_train.shape[1],1)) )
model.add(Flatten())
model.add(Dense(180, activation = 'relu'))
model.add(Dense(150, activation = 'relu'))
model.add(Dense(110, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))
model.summary()



# 3. 컴파일, 훈련
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['accuracy'])

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
    callbacks=[early_stopping,
#테스트용이니 모델체크는 하지 말자    model_check_point
    ])

model.save(model_save_path)
model.save_weights(weights_save_path)


# 4. 평가, 예측
result3 = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", result3[0])
print("accuracy: ", result3[1])

y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)
print("y_test:", y_test)
print("y_predict:", y_predict)



y_recovery = y_test # np.argmax(y_test, axis=1)

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




# 실습 
x_pred = np.array([97, 98, 99, 100]) # 이렇게 되어 있으면

# print(x_pred.shape)
x_pred = x_pred.reshape(x_pred.shape[0],1) # 이렇게 reshape하고

# print(x_pred.shape)

#원래 하던 reshape 하고 사용한다
x_pred = x_pred.reshape(x_pred.shape[1], x_pred.shape[0], 1) 


y_predict = model.predict(x_pred) # 평가 데이터 다시 넣어 예측값 만들기
# print("y_predict:\n", y)
print("y_predict:\n", y_predict)





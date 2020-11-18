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



modelpath = './model/keras53-2-{epoch:02d}-{val_loss:.4f}.hdf5'
model_save_path = "./save/keras53_2_cifar10_model.h5"
weights_save_path = './save/keras53_2_cifar10_weights.h5'

# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add( Conv2D(32, (3,3), padding='same', input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])) )
model.add(Dropout(0.2))

model.add( Conv2D(32, (3,3), padding='same', activation='relu') )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add( Conv2D(64, (3,3), padding='same', activation='relu') )
model.add( Conv2D(64, (3,3), padding='same', activation='relu') )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add( Conv2D(128, (3,3), padding='same', activation='relu') )
model.add( Conv2D(128, (3,3), padding='same', activation='relu') )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax') )
model.summary()



# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy',
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


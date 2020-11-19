import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


# 1.데이터
# 1.1 load_data
import numpy as np
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("x_train.shape:",x_train.shape)
print("y_train.shape:",y_train.shape)

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 1.2 train_test_split
# train과 test가 미리 나눠져 있으니 별도로 하지 않는다
# validation은 fit에서 별도로 split 한다


# 1.3 scaler
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


# 1.4 reshape
# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2],x_train.shape[3])
# x_test = x_test.reshape(x_test.shape[0],x_train.shape[1]*x_train.shape[2],x_train.shape[3])
# print("reshape x:", x_train.shape, x_test.shape)



modelpath = './model/keras61_2_{epoch:02d}_{val_loss:.4f}.hdf5'
model_save_path = "./save/keras61_2_cifar10_model.h5"
weights_save_path = './save/keras61_2_cifar10_weights.h5'


# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D

model = Sequential()
model.add(Conv1D(75, 3, input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])) )
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()


# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )


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

model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping,
    model_check_point
    ])

model.save(model_save_path)
model.save_weights(weights_save_path)



# 4. 평가, 예측

loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("accuracy: ", accuracy)


y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
print("y_test:\n", y_test)
print("y_predict:\n", y_predict)
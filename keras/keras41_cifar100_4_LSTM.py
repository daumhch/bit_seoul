import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
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




# reshape
# LSTM 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2],x_test.shape[3])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2],x_test.shape[3])
print("reshape x:", x_train.shape, x_test.shape)



# Scaler
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


# 10000개 데이터 중에 9000개는 validation, 1000개는 test
from sklearn.model_selection import train_test_split 
x_test,x_val, y_test,y_val = train_test_split(
    x_test,y_test, train_size=0.1, test_size=0.9)

print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)
print("after x_val.shape",x_val.shape)





# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2]) ))
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax') )
model.summary()


# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )


from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=10,
    mode='auto',
    verbose=2)


history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping])




# 4. 평가, 예측

loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("accuracy: ", accuracy)


y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
print("y_test:\n", y_test)
print("y_predict:\n", y_predict)



plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])

plt.title('val_accuracy')
plt.ylabel('val_accuracy, accuracy')
plt.ylabel('val_loss, loss')
plt.xlabel('epoch')

plt.legend(['val_accuracy','accuracy','val_loss','loss']) # 색인
plt.show()

print("cifar100 LSTM end")
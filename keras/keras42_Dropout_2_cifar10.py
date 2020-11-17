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



# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_train.shape[3])
print("reshape x:", x_train.shape, x_test.shape)


# Scaler
# 선택은 아무거나, 최적이라 생각하는 주관적 판단
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
# print(x_train[0])





# 10000개 데이터 중에 9980개는 validation, 20개는 test
# train_size=0.002, test_size=0.998
from sklearn.model_selection import train_test_split 
x_test,x_val, y_test,y_val = train_test_split(
    x_test,y_test, train_size=0.1, test_size=0.9)

print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)
print("after x_val.shape",x_val.shape)




from tensorflow.keras.layers import Dropout

# 2.모델
model = Sequential()
model.add( Conv2D(32, (3,3), padding='same', input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])) )
model.add( Conv2D(32, (1,1), padding='valid') )
model.add( Conv2D(32, (3,3), padding='same') )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add( Conv2D(64, (3,3), padding='same') )
model.add( Conv2D(64, (1,1), padding='valid') )
model.add( Conv2D(64, (3,3), padding='same') )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add( Conv2D(128, (3,3), padding='same') )
model.add( Conv2D(128, (1,1), padding='valid') )
model.add( Conv2D(128, (3,3), padding='same') )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(10, activation = 'softmax') )
model.summary()




# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy', #CNN은 웬만하면 categorical_crossentropy
    optimizer='adam',
    metrics=['accuracy'] # 이젠 accuracy를 관찰할만하다
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
# print("y_predict:\n", y)
print("y_test:\n", y_test)
print("y_predict:\n", y_predict)


#모델 시각
fig, loss_ax = plt.subplots()
 
acc_ax = loss_ax.twinx()
 
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
 
acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
 
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax. set_ylabel('accuracy')
 
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
 
plt.show()



print("keras42_Dropout_2_cifar10 end")




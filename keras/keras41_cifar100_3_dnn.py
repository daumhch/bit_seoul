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




# DNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_test.shape[3])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
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




# 2.모델
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],) ))
model.add(Dense(20, activation='relu') )
model.add(Dense(30, activation='relu') )
model.add(Dense(20, activation='relu') )
model.add(Dense(10, activation='relu') )
model.add(Dense(10, activation='relu') )
model.add(Dense(100, activation='softmax') )
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
    
    # 수치 계산이 아닌, 데이터 7만개 훈련이니, epoch이 그리 크지 않아도 되는 것...맞는지 확인해보자
    epochs=100, 
    batch_size=512,
    verbose=0, # 0=로그 출력하지 않기, 1=막대그래프, 2=손실 정보
    
    # 별도의 validation 데이터를 split 하지 않았으니, train에서 잘라 쓴다
    validation_split=0.2, 
    
    callbacks=[early_stopping]) 



# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=512)
print("loss: ", loss)
print("accuracy: ", accuracy)


y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
# print("y_predict:\n", y)
print("y_test:\n", y_test)
print("y_predict:\n", y_predict)



plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])

plt.title('loss & accuracy')
plt.ylabel('loss, accuracy')
plt.xlabel('epoch')

plt.legend(['loss', 'accuracy']) # 색인
plt.show()


print("cifar10 dnn end")





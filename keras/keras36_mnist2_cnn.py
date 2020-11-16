import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# OneHotEncoding



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("origin x shape:",x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print("origin y shape:",y_train.shape, y_test.shape) # (60000,) (10000,)

# print(x_train[0]) # 28x28 리스트 데이터
# print(y_train[0]) # 5







# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (60000,10) (10000,10)
print(y_train[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]




# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
# shape 뒷부분이 28, 28 일 필요는 없지만, 일단 그대로 가져다 사용한다
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print("reshape x:", x_train.shape, x_test.shape)



# Scaler
# 선택은 아무거나, 최적이라 생각하는 주관적 판단
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
# print(x_train[0])





# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add( Conv2D(32, (2,2), padding='same', input_shape=(28,28,1) ) )
model.add(MaxPooling2D(pool_size=(2,2) ))
model.add( Conv2D(64, (2,2) ) )
model.add(MaxPooling2D(pool_size=(2,2) ))
model.add(Flatten())
model.add(Dense(10, activation='softmax') )
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
    patience=100,
    mode='auto',
    verbose=2)


model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping]) # 0=로그 출력하지 않기, 1=막대그래프, 2=손실 정보



# 4. 평가, 예측

loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("accuracy: ", accuracy)




# 실습 1. test데이터를 10개 가져와서 predict을 만들 것
#         one hot encoding을 원복할 것
#         print('실제값:', ~~~~~)
#         print('예측값:', ~~~~~)
# 실습 2. 모델에 es, tensorboard 넣기




x_predict = x_train[10:20]

y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)
print("y_predict:", y_predict)

y_recovery = np.argmax(y_train, axis=1)
for cnt in range(10,20):
    print(y_recovery[cnt])


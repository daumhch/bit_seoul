import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
num_words = 10000
(x_train, y_train), (x_test, y_test) = reuters.load_data(
                                        num_words=num_words, test_split=0.2)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (8982,) (2246,)
# (8982,) (2246,)

print("x_train[0]:",x_train[0])
print("y_train[0]:",y_train[0])
# x_train[0]: [1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 
# 32, 132, 6, 109, 15, 17, 12]
# y_train[0]: 3

print("len x_train[0]:",len(x_train[0]))
print("len y_train[0]:",len(x_train[11]))
# len x_train[0]: 87
# len y_train[0]: 59


# y의 카테고리 갯수 출력
category = np.max(y_train)
print("카테고리:", category)
# 카테고리: 45

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print("y_bunpo:", y_bunpo)
print("y_bunpo.shape:",y_bunpo.shape)
# y_bunpo: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
# y_bunpo.shape: (46,)

# 실습
# 모델 구성

## 토큰은 필요 없을 것 같다

## OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


## 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 1000
pad_x_train = pad_sequences(x_train, padding='pre', maxlen=max_len) # pre=앞, post=뒤
pad_x_test = pad_sequences(x_test, padding='pre', maxlen=max_len) # pre=앞, post=뒤
# pad_x_train = pad_sequences(x_train, padding='pre') # pre=앞, post=뒤
# pad_x_test = pad_sequences(x_test, padding='pre') # pre=앞, post=뒤
print("pad_x_train:",pad_x_train)
print(pad_x_train.shape)



# 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
model = Sequential()
model.add(Embedding(num_words, y_bunpo.shape[0], 
                        input_length=pad_x_train.shape[1]))
model.add(Flatten()) # Flatten이 먹으려면 input_length를 정확히 입력해야 한다
model.add(Dense(y_bunpo.shape[0], activation='softmax'))
model.summary()



# 컴파일 훈련
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='auto',
    verbose=0)

model.fit(
    pad_x_train, y_train,
    epochs=30, 
    batch_size=512,
    verbose=0,
    validation_split=0.2,
    callbacks=[early_stopping]) 


# 평가 예측
acc = model.evaluate(pad_x_test, y_test, batch_size=512)[1]
print("acc:", acc)







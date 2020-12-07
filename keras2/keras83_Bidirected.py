import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기
os.environ["CUDA_VISIBLE_DEVICES"] = '1'



from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 영화 사이트 IMDB의 리뷰 데이터입니다. 
# 이 데이터는 리뷰에 대한 텍스트와 해당 리뷰가 
# 긍정인 경우 1을 부정인 경우 0으로 표시한 레이블로 
# 구성된 데이터입니다.

# imdb 소스를 완성하시오
num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (25000,) (25000,)
# (25000,) (25000,)

# print("x_train[0]:",x_train[0])
print("y_train[:10]:",y_train[:10])
# y_train[:10]: [1 0 0 1 0 0 1 0 1 0]

print("len x_train[0]:",len(x_train[0]))
print("len y_train[0]:",len(x_train[11]))
# len x_train[0]: 218
# len y_train[0]: 99


# y의 카테고리 갯수 출력
category = np.max(y_train) + 1
print("카테고리:", category)
# 카테고리: 1

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print("y_bunpo:", y_bunpo)
print("y_bunpo.shape:",y_bunpo.shape)
# y_bunpo: [0 1]
# y_bunpo.shape: (2,)



## 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 1000
x_train = pad_sequences(x_train, padding='pre', maxlen=max_len) # pre=앞, post=뒤
x_test = pad_sequences(x_test, padding='pre', maxlen=max_len) # pre=앞, post=뒤
# pad_x_train = pad_sequences(x_train, padding='pre') # pre=앞, post=뒤
# pad_x_test = pad_sequences(x_test, padding='pre') # pre=앞, post=뒤
print("x_train:",x_train)
print(x_train.shape)
# pad_x_train: [[    0     0     0 ...    19   178    32]     
#  [    0     0     0 ...    16   145    95]
#  [    0     0     0 ...     7   129   113]
#  ...
#  [    0     0     0 ...     4  3586 22459]
#  [    0     0     0 ...    12     9    23]
#  [    0     0     0 ...   204   131     9]]
# (25000, 1000)


# 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional
model = Sequential()
model.add(Embedding(num_words, 64))
# model.add(LSTM(64))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))
model.summary()




# 컴파일 훈련
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=5,
    mode='auto',
    verbose=0)

model.fit(
    x_train, y_train,
    epochs=30, 
    callbacks=[early_stopping]) 


# 평가 예측
acc = model.evaluate(x_test, y_test)[1]
print("acc:", acc)


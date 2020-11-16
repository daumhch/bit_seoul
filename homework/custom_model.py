import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

import numpy as np

# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import BatchNormalization
model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(3,1) ) ) 
model.add(BatchNormalization())
model.add(Dense(30))
model.add(BatchNormalization())
model.add(Dense(30))
model.add(BatchNormalization())
model.add(Dense(20))
model.summary()
# total params: 2540

model.save("./save/custom_model.h5") # root/save 폴더 밑에 저장

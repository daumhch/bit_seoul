import numpy as np

# 2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

model = Sequential()
model.add( LSTM(75, activation='relu', input_shape=(4,1) ) )
model.add(Dense(180))
model.add(Dense(150))
model.add(Dense(110))
model.add(Dense(60))
model.add(Dense(10))

model.summary()

# model.save("save1.h5") # root에 저장
model.save("./save/keras28.h5") # root/save 폴더 밑에 저장

# 경로는 아래와 같이 다 가능하다
model.save(".\save\keras28_2.h5")
model.save(".//save//keras28_3.h5")
model.save(".\\save\\keras28_4.h5")

# 다만, 경로 내에 \n 하면 개행문자로 읽을 수도 있으니,
# \는 되도록 하지말자(내 생각)



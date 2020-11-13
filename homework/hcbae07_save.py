import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 2.모델
model = Sequential()
model.add(LSTM(100, name="LSTM", input_shape=(4,1)))
model.add(Dense(50, name="Dense1"))
model.add(Dense(10, name="Dense2"))
model.add(Dense(1, name="Dense3"))

model.summary()

# model.save("save1.h5") # root에 저장
model.save("./save/hcbae_model_save.h5") # root/save 폴더 밑에 저장

# 경로는 아래와 같이 다 가능하다
# model.save(".\save\keras28_2.h5")
# model.save(".//save//keras28_3.h5")
# model.save(".\\save\\keras28_4.h5")

# 다만, 경로 내에 \n 하면 개행문자로 읽을 수도 있으니,
# \는 되도록 하지말자(내 생각)



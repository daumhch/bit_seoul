# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Input

input1 = Input(shape=(3,1))
lstm1 = GRU(20, activation='relu')(input1)
dense = Dense(180, activation='relu')(lstm1)
dense = Dense(10, activation='relu')(dense)
output1 = Dense(1)(dense)
model = Model(inputs=input1, outputs=output1)

model.summary()

model.save("./save/hcbae_model_save2.h5") # root/save 폴더 밑에 저장

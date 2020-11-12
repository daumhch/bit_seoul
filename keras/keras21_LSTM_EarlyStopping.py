import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np

x = np.array([
    [1,2,3],[2,3,4],[3,4,5],[4,5,6],
    [5,6,7],[6,7,8],[7,8,9],[8,9,10],
    [9,10,11],[10,11,12],
    [20,30,40],[30,40,50],[40,50,60]
    ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print("x.shape:", x.shape)
print("y.shape:", y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1) # (4,3,1)
print("reshape x.shape:", x.shape)

x_input = np.array([50,60,70])
x_input = x_input.reshape(1,3,1)

# 실습 LSTM 완성하시오
# 예측값 80

# 함수형으로 코딩하시오


# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Input
'''
model = Sequential()
model.add(GRU(200, activation='relu', input_shape=(3,1) ))
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
'''
input1 = Input(shape=(3,1))
lstm1 = GRU(200, activation='relu')(input1)
dense1 = Dense(180, activation='relu')(lstm1)
dense2 = Dense(150, activation='relu')(dense1)
dense3 = Dense(110, activation='relu')(dense2)
dense4 = Dense(60, activation='relu')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

# model.summary()




# 3. 컴파일, 훈련
model.compile( # 컴파일
    loss='mse', # 오차함수는 mean squared error를 사용한다
    optimizer='adam') # 최적화 방법은 'adam'을 사용한다

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
# early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min')
# 지표가 min max인지 모른다면 auto로 놓자
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto', baseline=0.01)
model.fit(x, y,
    epochs=10000, # 훈련 횟수
    batch_size=1, # 훈련 데이터단위
    verbose=1,
    callbacks=[early_stopping]) # 훈련하고 호출하기, 리스트인걸 봐선 여러개 가능할 듯




# 4. 평가, 예측
# 평가 데이터 넣어서 결과 보기
result = model.evaluate(x, y, batch_size=1) 
print("result : ", result)


y_predict = model.predict(x_input) # 평가 데이터 다시 넣어 예측값 만들기
print("y_predict:\n", y_predict)



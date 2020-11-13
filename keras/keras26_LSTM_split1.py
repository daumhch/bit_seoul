import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



import numpy as np
dataset = np.array(range(1,11))
size = 5



# 모델을 구성하시오(fit까지)
def hcbae_split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        # [item~~~] 이 없어도 되는데, 있는 이유가 있겠지?
        # aaa.insert(i, subset)
        aaa.append(subset)
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자


datasets = hcbae_split_x(dataset,size)
# print(datasets)
# print(datasets.shape)

x = datasets.T[0:(datasets.shape[1]-1)]
y = datasets.T[(datasets.shape[1]-1):(datasets.shape[1])]
# print(x)
# print(y)

x = x.T
y = y.T
# print(x)
# print(y)

print("x.shape:", x.shape)
x = x.reshape(x.shape[0], x.shape[1], 1) # (4,3,1)
print("reshape x.shape:", x.shape)





# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape=(x.shape[1],1))
lstm = LSTM(200, activation='relu')(input1)
dense = Dense(180, activation='relu')(lstm)
dense = Dense(150, activation='relu')(dense)
dense = Dense(110, activation='relu')(dense)
dense = Dense(60, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output1 = Dense(1)(dense)
model = Model(inputs=input1, outputs=output1)

# model.summary()



# 3. 컴파일, 훈련
model.compile( # 컴파일
    loss='mse', # 오차함수는 mean squared error를 사용한다
    optimizer='adam') # 최적화 방법은 'adam'을 사용한다

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=100,
    mode='auto',
    verbose=2)

history = model.fit(x, y, #훈련, 일단 x_train, y_train 입력하고
    epochs=10000, # 훈련 횟수
    verbose=0,
    callbacks=[early_stopping]) # 0=로그 출력하지 않기, 1=막대그래프, 2=손실 정보


# 4. 평가, 예측
# 평가 데이터 넣어서 결과 보기
result = model.evaluate(x, y) 
print("result: ", result) # 이건 기본으로 나오고


# 실습 7, 8, 9, 10에 대한 predict 하기
x_test = np.array([[7, 8, 9, 10],[8, 9, 10, 11]])
# print(x_test.shape)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) # (4,3,1)
y_test = [11, 12]
y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
# print("y_predict:\n", y)
print("y_predict:\n", y_predict)




# 사용자정의 RMSE 함수
# 사이킷런의 metrics에서 mean_squared_error을 불러온다
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_test, y_predict))





# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2:", r2)




# 그래프 그리기
import matplotlib.pyplot as pyplot

# plot train and validation loss
pyplot.plot(history.history['loss'])
# pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.ylim((0,1000))
pyplot.show()






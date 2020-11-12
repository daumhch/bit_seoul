# keras12~keras25
# 데이터 전처리 전 까지 모든 파트를 반영한 소스

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# multi layer perceptron
# 1.데이터
import numpy as np # numpy를 불러오고 앞으로 이름을 np로 쓴다
x1 = np.array([
    [1,2,3],[2,3,4],[3,4,5],[4,5,6],
    [5,6,7],[6,7,8],[7,8,9],[8,9,10],
    [9,10,11],[10,11,12],
    [20,30,40],[30,40,50],[40,50,60]
    ])
x2 = np.array([
    [10,20,30],[20,30,40],[30,40,50],[40,50,60],
    [50,60,70],[60,70,80],[70,80,90],[80,90,100],
    [90,100,110],[100,110,120],
    [2,3,4],[3,4,5],[4,5,6]
    ])
x3 = np.array([
    [110,120,130],[120,130,140],[130,140,150],[140,150,160],
    [150,160,170],[160,170,180],[170,180,190],[180,190,1100],
    [190,1100,1110],[1100,1110,1120],
    [12,13,14],[13,14,15],[14,15,16]
    ])
y = np.array([
    4,5,6,7,
    8,9,10,11,
    12,13,
    50,60,70])


def hcbae_split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        # [item~~~] 이 없어도 되는데, 있는 이유가 있겠지?
        # aaa.insert(i, subset)
        aaa.append(subset)
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자
# end of hcbae_split_x


array_size = 3

# data_size당 훈련 1회 시간
# 10000 = 2~3초
# 5000 = 1~2초
# 1000 = 13ms
data_size = 5000

x1 = hcbae_split_x(range(1,1+data_size+array_size-1),array_size)
x2 = hcbae_split_x(range(101,101+data_size+array_size-1),array_size)
x3 = hcbae_split_x(range(201,201+data_size+array_size-1),array_size)

y = np.array(range(1,data_size+1))




print("before x1.shape",x1.shape)
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) # (13,3,1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) # (13,3,1)
x3 = x3.reshape(x3.shape[0], x3.shape[1], 1) # (13,3,1)
print("after x1.shape",x1.shape)
print("after x2.shape",x2.shape)
print("after x3.shape",x3.shape)
print("before y.shape",y.shape)


# 사이킷런의 model_selection에서 train_test_split을 불러온다
from sklearn.model_selection import train_test_split 

# 데이터는 train / val / test로 나뉘고 비율은 6:2:2를 보통 쓴다
# train_test_split를 사용하면 랜덤하게 뽑아서 나눠준다
x1_train,x1_rest, x2_train,x2_rest = train_test_split(
    x1, x2, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고

x1_test,x1_val, x2_test,x2_val = train_test_split(
    x1_rest, x2_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔


x3_train,x3_rest, y_train,y_rest = train_test_split(
    x3, y, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고

x3_test,x3_val, y_test,y_val = train_test_split(
    x3_rest, y_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔




# 2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, SimpleRNN, GRU

input_LSTM = Input(shape=(3,1))
dense_LSTM = LSTM(32, activation='relu')(input_LSTM)
dense_LSTM = Dense(512, activation='relu')(dense_LSTM)
output_LSTM = Dense(256)(dense_LSTM)

input_SRNN = Input(shape=(3,1))
dense_SRNN = SimpleRNN(32, activation='relu')(input_SRNN)
dense_SRNN = Dense(512, activation='relu')(dense_SRNN)
output_SRNN = Dense(256)(dense_SRNN)

input_GRU = Input(shape=(3,1))
dense_GRU = GRU(32, activation='relu')(input_GRU)
dense_GRU = Dense(512, activation='relu')(dense_GRU)
output_GRU = Dense(256)(dense_GRU)

from tensorflow.keras.layers import concatenate
merge_model = concatenate([output_LSTM, output_SRNN, output_GRU])
merge_temp = Dense(512, name='merge_temp1')(merge_model)
merge_temp = Dense(256, name='merge_temp2')(merge_temp)
merge_temp = Dense(128, name='merge_temp3')(merge_temp)
merge_temp = Dense(64, name='merge_temp4')(merge_temp)
output_total = Dense(1)(merge_temp)

# 모델 정의
model = Model(inputs=[input_LSTM, input_SRNN, input_GRU], 
                outputs=output_total)
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

hist = model.fit( [x1_train, x2_train, x3_train], y_train, #훈련, 일단 x_train, y_train 입력하고
    epochs=10000, # 훈련 횟수
    batch_size=32, # 훈련 데이터단위
    validation_data=([x1_val, x2_val, x3_val], y_val), # 검증 데이터 사용하기
    verbose=1,
    callbacks=[early_stopping]) # 0=로그 출력하지 않기, 1=막대그래프, 2=손실 정보





# 4. 평가, 예측
# 평가 데이터 넣어서 결과 보기
result = model.evaluate([x1_test, x2_test, x3_test], y_test, batch_size=32) 
print("result: ", result) # 이건 기본으로 나오고


y_predict = model.predict([x1_test, x2_test, x3_test]) # 평가 데이터 다시 넣어 예측값 만들기
# print("x_test array:", [x1_test, x2_test, x3_test])
print("y_predict:\n", y_test)
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

# print("x1_train.shape", x1_train.shape)
# print("x1_val.shape", x1_val.shape)
# print("x1_test.shape", x1_test.shape)


# 그래프 그리기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()




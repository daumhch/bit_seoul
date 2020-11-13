import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



import numpy as np
dataset = np.array(range(1,101))
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
# x = x.reshape(x.shape[0], x.shape[1], 1) # (4,3,1)
# print("reshape x.shape:", x.shape)



# 사이킷런의 model_selection에서 train_test_split을 불러온다
from sklearn.model_selection import train_test_split 
x_train,x_rest, y_train,y_rest = train_test_split(
    x, y, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x_test,x_val, y_test,y_val = train_test_split(
    x_rest,y_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔

print("x_train x.shape:", x_train.shape)

print("y_train.shape", y_train.shape)
print("y_val.shape", y_val.shape)
print("y_test.shape", y_test.shape)



# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape=(4,))
dense = Dense(200, activation='relu')(input1)
dense = Dense(180, activation='relu')(dense)
dense = Dense(150, activation='relu')(dense)
dense = Dense(110, activation='relu')(dense)
dense = Dense(60, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output1 = Dense(1)(dense)
model = Model(inputs=input1, outputs=output1)

model.summary()



# 3. 컴파일, 훈련
model.compile( # 컴파일
    loss='mse', # 오차함수는 mean squared error를 사용한다
    optimizer='adam',
    metrics=['mse']) # 최적화 방법은 'adam'을 사용한다

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=100,
    mode='auto',
    verbose=2)

history = model.fit(x_train, y_train, #훈련, 일단 x_train, y_train 입력하고
    epochs=10000, # 훈련 횟수
    verbose=0,
    validation_data=(x_val,y_val),
    callbacks=[early_stopping]) # 0=로그 출력하지 않기, 1=막대그래프, 2=손실 정보


# 4. 평가, 예측
# 평가 데이터 넣어서 결과 보기
loss, mse = model.evaluate(x, y)
print("loss: ", loss) # 이건 기본으로 나오고
print("mse: ", mse)


# 실습 
x_pred = np.array([97, 98, 99, 100]) # 이렇게 되어 있으면
# print(x_pred.shape)
x_pred = x_pred.reshape(1, x_pred.shape[0]) # 이번엔 이렇게 reshape하고
# print(x_pred.shape)

#원래 하던 reshape 하고 사용한다
# x_pred = x_pred.reshape(x_pred.shape[1], x_pred.shape[0], 1) 
print("x_pred.shape:", x_pred.shape)

y_predict = model.predict(x_pred) # 평가 데이터 다시 넣어 예측값 만들기
# print("y_predict:\n", y)
print("y_predict:\n", y_predict)





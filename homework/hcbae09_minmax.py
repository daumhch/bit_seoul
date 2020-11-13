import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Input

#1. 데이터
dataset = np.array(range(1,105))
size = 5


def hcbae_split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        # [item~~~] 이 없어도 되는데, 있는 이유가 있겠지?
        # aaa.insert(i, subset)
        aaa.append(subset)
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자


datasets = hcbae_split_x(dataset,size)

x = datasets[:, :4]
y = datasets[:, 4]
print("x.shape:", x.shape) # (96,4)



# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
# print(x)
scaler = MinMaxScaler()
scaler.fit(x) # fit하고
x = scaler.transform(x) # 사용할 수 있게 바꿔서 저장하자
# print(x)



x = np.reshape(x, (x.shape[0], x.shape[1], 1))
print("x.reshape:", x.shape) # (96,4,1)


from sklearn.model_selection import train_test_split 
x_train,x_rest, y_train,y_rest = train_test_split(
    x, y, shuffle=False, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x_val, x_test, y_val, y_test = train_test_split(
    x_rest,y_rest, shuffle=False, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔

print("x_train.reshape:", x_train.shape) # (57,4,1)




# 2.모델
model = load_model("./save/keras28.h5")
model(Input(shape=(4,1), name='input1'))
model.add(Dense(1, name='output1'))
# model.summary()



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

y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
# print("y_predict:\n", y)
print("y_test:\n", y_test)
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




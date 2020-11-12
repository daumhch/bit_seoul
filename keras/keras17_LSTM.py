import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


# 1.데이터
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) # 4행 3열 = (4, 3)
y = np.array([4,5,6,7]) # (4, )
# 1,2,3으로 4를 예측하겠다

print("x.shape:", x.shape)
print("y.shape:", y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1) # (4,3,1)

print("reshape x.shape:", x.shape)




# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(256, activation='relu', input_shape=(3,1)))
# input_shape가 (3,1)인 이유 = LSTM에서는 몇 개 씩 명시해줘야 한다
# LSTM은 행무시하고 2차원 shape를 받는다
# (3,1)은, 1개씩 잘라서 연산한다는 의미
# LSTM의 shape는, 행무시 열우선 몇개씩 자를 것인지 정해야 한다
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(1))

model.summary()




# 3. 컴파일, 훈련
model.compile( # 컴파일
    loss='mse', # 오차함수는 mean squared error를 사용한다
    optimizer='adam', # 최적화 방법은 'adam'을 사용한다
    metrics=['mae']) # mean absolute error 지표를 추가한다

#데이터가 짧으니 validation은 하지 말자
model.fit(x, y, 
    epochs=512, # 훈련 횟수
    batch_size=1, # 훈련 데이터단위
    verbose=0)




# 4. 평가, 예측
# 평가 데이터 넣어서 결과 보기
result = model.evaluate(x, y, batch_size=1) 
print("result : ", result)

x_input = np.array([5,6,7]) # (3,) -> (1,3,1)로 바꿔야 한다
x_input = x_input.reshape(1,3,1)

y_predict = model.predict(x_input) # 평가 데이터 다시 넣어 예측값 만들기
print("y_predict:\n", y_predict)


'''
# 사용자정의 RMSE 함수
# 사이킷런의 metrics에서 mean_squared_error을 불러온다
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y, y_predict))



# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print("R2:", r2)
'''




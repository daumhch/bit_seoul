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
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

model = Sequential()
model.add(SimpleRNN(30, activation='relu', input_shape=(3,1)))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(1))

model.summary()


'''
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



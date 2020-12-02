import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # Dense = DNN구성
import numpy as np

# 1. 데이터 준비
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([11,12,13,14,15])


# 2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim=1, activation='sigmoid'))
model.add(Dense(5000, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(7, activation='sigmoid'))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(
loss='mse',
optimizer='adam',
metrics=['acc'])

model.fit(x, y, epochs=100, verbose=0)


# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)


y_predict = model.predict(x2)
print("y_predict:\n", y_predict)



########## sigmoid를 다 넣으면 0과 1사이로 수렴한게 전달되기 때문에
########## 정확한 예측을 할 수 없다
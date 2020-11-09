import numpy as np

# 1. 데이터 준비
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # Dense = DNN구성


# 2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim=1)) # node 300 layer 1
model.add(Dense(5000)) # node 5000 layer 2
model.add(Dense(30)) # node 30 layer 3
model.add(Dense(7)) # node 30 layer 3
model.add(Dense(1)) # output


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# mse = mean squared error
# adam = 왠만하면 adam, 최적화 기법.
# acc = 평가지표는 정확도.

model.fit(x, y, epochs=100, batch_size=1) 
# 데이터를 1개씩 잘라서 100번 훈련하자


# 4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)

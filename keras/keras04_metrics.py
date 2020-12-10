
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # Dense = DNN구성
import numpy as np

# 1. 데이터 준비
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# x2 = np.array([11,12,13,14,15])
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)


# 2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=1))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(1)) # node=1 last_layer=output


# 3. 컴파일, 훈련
model.compile(
loss='mse',
optimizer='adam')
# mse = mean squared error
# adam = 왠만하면 adam, 최적화 기법.
# acc = 평가지표는 정확도.

# model.fit(x, y, epochs=100, batch_size=1)
model.fit(x, y, epochs=128) # batch_size는 default가 32, 조정 전 기준
# 데이터를 1개씩 잘라서 100번 훈련하자


# 4. 평가, 예측
#loss, acc, accuracy, categorical_accuracy = model.evaluate(x, y, batch_size=1)
loss = model.evaluate(x, y)
print("loss : ", loss)

y_predict = model.predict([10])
print("y_predict:\n", y_predict)




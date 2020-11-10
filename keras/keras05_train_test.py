from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터 준비
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11,12,13])


# 2. 모델 구성
model = Sequential()
model.add(Dense(1024, input_dim=1))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(
loss='mse',
optimizer='adam',
metrics=['mae'])

model.fit(x, y, epochs=100, batch_size=5)


# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=5)
print("loss : ", loss)

y_predict = model.predict(x_pred)
print("y_predict:\n", y_predict)




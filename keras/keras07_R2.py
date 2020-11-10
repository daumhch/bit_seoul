from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터 준비
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])


# 2. 모델 구성
model = Sequential()
model.add(Dense(256, input_dim=1))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(
loss='mse',
optimizer='adam',
metrics=['mae'])

model.fit(x_train, y_train, epochs=512, batch_size=32)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)

# y_predict = model.predict(x_pred) # 단순 평가
y_predict = model.predict(x_test) # 예측값을 평가하기 위해 따로
print("y_predict:\n", y_predict)


# 사용자정의 RMSE 함수
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_test, y_predict))


# 사용자정의 R2 함수
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2:", r2)





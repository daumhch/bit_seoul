# 1.데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))

from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3)
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.3, shuffle=False)

print(x_test)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(256, input_dim=1))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae'])

model.fit(x_train, y_train, epochs=256, batch_size=32, 
    validation_split=0.2)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)

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



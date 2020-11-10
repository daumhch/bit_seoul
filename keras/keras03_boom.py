
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # Dense = DNN구성
import numpy as np

# 1. 데이터 준비
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([11,12,13,14,15])


# 2. 모델 구성
model = Sequential()
# 아래와 같은 설정을 하면 GPU 메모리가 터진다 = 에러나고 끝난다
model.add(Dense(3000000, input_dim=1))
model.add(Dense(5000000))
model.add(Dense(90000))
model.add(Dense(30000))
model.add(Dense(1)) # node=1 last_layer=output


# 3. 컴파일, 훈련
model.compile(
loss='mse',
optimizer='adam',
metrics=['acc',
'accuracy'])
# mse = mean squared error
# adam = 왠만하면 adam, 최적화 기법.
# acc = 평가지표는 정확도.

# model.fit(x, y, epochs=100, batch_size=1)
model.fit(x, y, epochs=10000) # batch_size는 default가 32, 조정 전 기준
# 데이터를 1개씩 잘라서 100번 훈련하자


# 4. 평가, 예측
#loss, acc, accuracy, categorical_accuracy = model.evaluate(x, y, batch_size=1)
loss, acc, accuracy = model.evaluate(x, y)
print("loss : ", loss)
print("acc : ", acc)
print("accuracy : ", accuracy)

y_predict = model.predict(x2)
print("y_predict:\n", y_predict)




import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


# keras02 -> keras 68
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # Dense = DNN구성
import numpy as np



# 1. 데이터 준비
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([11,12,13,14,15])


# 2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=1))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(1)) # node=1 last_layer=output

# 7개
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.001)     # result: 0.001093382015824318
# optimizer = Adadelta(lr=0.001) # result: 31.56778335571289
# optimizer = Adamax(lr=0.001)   # result: 0.003606173675507307
# optimizer = Adagrad(lr=0.001)  # result: 0.0002795398759189993
# optimizer = RMSprop(lr=0.001)    # result: 0.00010631301847752184
# optimizer = SGD(lr=0.001)      # result: 0.38846355676651
optimizer = Nadam(lr=0.001)    # result: 0.008073488250374794


# 3. 컴파일, 훈련
model.compile(
loss='mse',
optimizer=optimizer)

model.fit(x, y, epochs=10, batch_size=1)


# 4. 평가, 예측
result = model.evaluate(x, y, batch_size=1)
print("result : ", result)


y_predict = model.predict(x2)
print("result:",result," y_predict:\n", y_predict)




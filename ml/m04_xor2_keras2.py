# from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1.데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]


# 2.모델
# model = LinearSVC()
# model = SVC()
model = Sequential()
model.add(Dense(8, input_shape=(2,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3.훈련
# model.fit(x_data, y_data)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_data,y_data, batch_size=1, epochs=100)


# 4.평가 및 예측
y_predict = model.predict(x_data)
print("y_predict:\r\n", y_predict)

# score = model.score(x_data, y_data)
score = model.evaluate(x_data, y_data)
print("score:\r\n", score)

import numpy as np
y_predict = np.round(y_predict, 0)
print("y_predict:\r\n", y_predict)

acc = accuracy_score(y_data, y_predict)
print("acc:\r\n", acc)






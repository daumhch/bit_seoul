
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.array([1,2,3,4,5,6])
y = np.array([2,4,6,8,10,12])
x2 = np.array([11,12,13,14])


model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(512))
model.add(Dense(64))
model.add(Dense(1))


model.compile(
loss='mse',
optimizer='adam',
metrics=['acc',
'accuracy',
'categorical_accuracy'])


model.fit(x, y, epochs=100, batch_size=2)

loss, acc, accuracy, categorical_accuracy = model.evaluate(x, y, batch_size=2)
print("loss : ", loss)
print("acc : ", acc)
print("accuracy : ", accuracy)
print("categorical_accuracy : ", categorical_accuracy)


y_predict = model.predict(x2)
print("y_predict:", y_predict)

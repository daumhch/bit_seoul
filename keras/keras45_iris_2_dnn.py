import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# 다중분류
import numpy as np
from sklearn.datasets import load_iris

# 꽃 잎과 꽃 줄기를 보고 꽃을 판단하는 것
datasets = load_iris()
x = datasets.data
y = datasets.target

print("x.shape:",x.shape)
print("y.shape:",y.shape)

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)




# train/test/val
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1)

print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)



# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) # fit하고
x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자
x_test = scaler.transform(x_test) # 사용할 수 있게 바꿔서 저장하자





# reshape?
# x가 2차원이라, 별도의 reshape 없이 그대로 x를 사용한다




# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],) ))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()




# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])



from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='accuracy',
    patience=50,
    mode='auto',
    verbose=2)

history = model.fit(x_train, y_train,
    epochs=500,
    verbose=0,
    validation_split=0.2,
    callbacks=[early_stopping],
    batch_size=128)



# 4. 평가 및 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("accuracy: ", accuracy)


y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
print("y_test:\n", y_test)
print("y_predict:\n", y_predict)



# 사용자정의 RMSE 함수
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_test, y_predict))



# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2:", r2)




import matplotlib.pyplot as plt
#모델 시각
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
acc_ax.plot(history.history['accuracy'], 'b', label='train accuracy')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val accuracy')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax. set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
 
plt.show()
model.summary()




print("keras45_iris_1_cnn end")





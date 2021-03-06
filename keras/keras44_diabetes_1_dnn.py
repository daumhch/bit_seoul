import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


# 당뇨병 관련 데이터


'''
x
442 행 10 열 
Age
Sex
Body mass index
Average blood pressure
S1
S2
S3
S4
S5
S6

y
442 행 1 열 
target: a quantitative measure of disease progression one year after baseline
'''


import numpy as np
from sklearn.datasets import load_diabetes
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print("x.shape:",x.shape) # (506, 13)
print("y.shape:",y.shape) # (506, )
print("y[0:10]:", y[0:10])

# 구현 순서
# 데이터 전처리
# 모델
# 컴파일 훈련
# 평가 예측
# 회귀는 RMSE와 R2

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x) # fit하고
x = scaler.transform(x) # 사용할 수 있게 바꿔서 저장하자





# train/test/val
from sklearn.model_selection import train_test_split 
x_train,x_rest, y_train,y_rest = train_test_split(
    x, y, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x_test,x_val, y_test,y_val = train_test_split(
    x_rest,y_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔

print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)
print("after x_val.shape",x_val.shape)



# reshape?
# x가 2차원이라, 별도의 reshape 없이 그대로 x를 사용한다



# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],) ))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1) )
model.summary()


# 3. 컴파일, 훈련
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae'])


from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=100,
    mode='auto',
    verbose=2)

history = model.fit(x_train, y_train,
    epochs=10000,
    verbose=0,
    validation_data=(x_val,y_val),
    callbacks=[early_stopping],
    batch_size=1)



# 4. 평가 및 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print("loss: ", loss) # 이건 기본으로 나오고
print("mae: ", mae)



y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
print("y_predict:\n", y_test)
print("y_predict:\n", y_predict)
print("y_predict.shape:\n", y_predict.shape)

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
 
acc_ax.plot(history.history['mae'], 'b', label='train mae')
acc_ax.plot(history.history['val_mae'], 'g', label='val mae')
 
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax. set_ylabel('mae')
 
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
 
plt.show()
model.summary()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print("x.shape:",x.shape) # (506, 13)
print("y.shape:",y.shape) # (506, )

# 중복을 제거한 후, 종류를 살펴서 회귀인지 분류인지 판단할 수도 있겠다
# set(y)는 중복 제거한 집합 표시
# len(set(y))는 중복 제거한 집합의 길이
print("중복 제거한 y의 내부 종류:", set(y) )
print("중복 제거한 y의 길이:",len(set(y)) )

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)


# train_test_split
# scaler
# reshape

# train_test_split
# train/test/val
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1)

print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)




# Scaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) # fit하고
x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자
x_test = scaler.transform(x_test) # 사용할 수 있게 바꿔서 저장하자




# reshape?
# x가 2차원이라, 별도의 reshape 없이 그대로 x를 사용한다





# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(int(x_train.shape[1]/1), activation='relu', input_shape=(x_train.shape[1],) ))
model.add(Dense(int(x_train.shape[1]/2), activation='relu'))
model.add(Dense(int(x_train.shape[1]/3), activation='relu'))
model.add(Dense(int(x_train.shape[1]/4), activation='relu'))
model.add(Dense(int(x_train.shape[1]/5), activation='relu'))
model.add(Dense(int(x_train.shape[1]/6), activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.summary()






# 3. 컴파일, 훈련
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=5,
    mode='auto',
    verbose=2)

from tensorflow.keras.callbacks import ModelCheckpoint # 모델 체크 포인트
modelpath = './model/cancer-{epoch:02d}-{val_loss:.4f}.hdf5'
model_check_point = ModelCheckpoint(
    filepath=modelpath,
    monitor='val_loss',
    save_best_only=True,
    mode='auto')

hist = model.fit(
    x_train, y_train,
    epochs=100, 
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping, model_check_point]) 



# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("mae: ", mae)


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




# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) # 단위는 찾아보자

plt.subplot(2,1,1) # 2장 중에 첫 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2) # 2장 중에 두 번째
plt.plot(hist.history['mae'], marker='.', c='red')
plt.plot(hist.history['val_mae'], marker='.', c='blue')
plt.grid()
plt.title('mae')
plt.ylabel('mae')
plt.xlabel('epochs')
plt.legend(['mae', 'val_mae'])

plt.show()


print("keras49_ModelCheckPoint_3_cifar100 end")



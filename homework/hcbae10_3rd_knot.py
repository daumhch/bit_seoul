# keras26~keras34

# 정리할 것
# 모델 로드 + 텐서보드
# 스케일러는 텐서보드 다음에

# hcbae_06_2nd_knot을 발전시켜서 매듭을 만든다

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# 1.데이터
import numpy as np
array_size = 10
data_size = 100

# def도 다른 파일에서 import하자 매번 선언하기 귀찮다
from custom_def import hcbae_split_x 
datasets = hcbae_split_x(range(10000,10000+data_size+array_size-1),array_size)
# 위처럼 하면, 
# 연속된 데이터를 array_size만큼 잘라서
# data_size 길이만큼 만든다
# 즉, data_size=100, array_size=5 설정하면, shape = (100,5)
print("datasets.shape:", datasets.shape)


# x는 여러개, y는 array_size의 마지막 1개로 하자
x = datasets[:,:(array_size-1)]
y = datasets[:,(array_size-1)]
print("x.shape:", x.shape)
print("y.shape:", y.shape)


# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
# print(x)
# scaler = MinMaxScaler()
# scaler = RobustScaler()
scaler = StandardScaler()
scaler.fit(x) # fit하고
x = scaler.transform(x) # 사용할 수 있게 바꿔서 저장하자
# print(x)


# LSTM 모델 input을 위한 reshape
x = x.reshape(x.shape[0], x.shape[1], 1)
print("x.reshape:", x.shape) # (96,4,1)


from sklearn.model_selection import train_test_split 
x_train,x_rest, y_train,y_rest = train_test_split(
    x, y, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x_test,x_val, y_test,y_val = train_test_split(
    x_rest,y_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔

print("x_train.reshape:", x_train.shape) # (57,4,1)






# 모델을 무엇을 쓸 것인가에 따라서
# reshape가 필요할 수도 있다
# LSTM계열은 3차원 입력을 받기 때문에
# 예를 들어, (5,4)데이터를 (5,4,자르는 갯수) shape로 바꿔야 한다

# 일단은 무난한 Dense 모델을 불러오자
# 를 해봤는데, 순수 Dense로 만들고 input_shape=(2,1)로 했는데,
# 자꾸 에러가 난다
# ValueError: Input 0 of layer sequential is incompatible with the layer: 
# expected axis -1 of input shape to have value 2 but received input with shape [None, 4, 1]
# 그래서 LSTM 모델로 저장하고, LSTM을 불러오자
# LSTM 모델이라 하더라도, 저장된 모델의 shape와, 불러와서 사용하는 shape가 다르면 warning이 뜬다


from tensorflow.keras.models import load_model
lstm_model = load_model("./save/custom_model.h5")

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
lstm_model(Input(shape=(x.shape[1],1), name='input1'))
lstm_model.add(Dense(1, name='output1'))
# lstm_model.summary()





# 3. 컴파일, 훈련
lstm_model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=100,
    mode='auto',
    verbose=2)

from tensorflow.keras.callbacks import TensorBoard # 텐서보드 = 웹페이지 시각화

from custom_def import hcbae_removeAllFile
hcbae_removeAllFile("./graph/") # graph폴더 지우고 새로 만듬

to_tensorboard = TensorBoard(
    log_dir='graph',
    histogram_freq=0,
    write_graph=True,
    write_images=True)

history = lstm_model.fit(x_train, y_train,
    epochs=10000,
    verbose=0,
    validation_data=(x_val,y_val),
    callbacks=[early_stopping, to_tensorboard])

# 위와 같이 tensorboard를 추가하고,
# cmd창에서, graph폴더에 들어간 후, tensorboard --logdir=. 를 실행하면
# http://localhost:6006/ 주소에서 결과를 확인할 수 있다




loss, mse = lstm_model.evaluate(x_test, y_test)
print("loss: ", loss) # 이건 기본으로 나오고
print("mse: ", mse)


y_predict = lstm_model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
print("y_predict:\n", y_test)
print("y_predict:\n", y_predict)


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_test, y_predict))



# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2:", r2)


# history
print("=======history start=======")
print(history.history.keys())
print("=======history end=======")


# 그래프 그리기 = 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])

plt.title('loss & mae')
plt.ylabel('loss, mae')
plt.xlabel('epoch')

plt.legend(['train loss', 'val loss', 'train mae', 'val mae']) # 색인
plt.show()




import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


from numpy import array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

# 1.데이터
x = array(
    [[1,2,3],[2,3,4],[3,4,5],[4,5,6],
    [5,6,7],[6,7,8],[7,8,9],[8,9,10],
    [9,10,11],[10,11,12],
    [2000,3000,4000],[3000,4000,5000],[4000,5000,6000],
    [100,200,300]])
y = array(
    [4,5,6,7,
    8,9,10,11,
    12,13,
    5000,6000,7000,
    400]) # (14,3)

x_predict = array([55,65,75]) # array([55,65,75]) # (3,)
x_predict2 = array([6600,6700,6800]) # (3,)

x_predict = x_predict.reshape(1,3)


# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
# print(x)
scaler = MinMaxScaler()
scaler.fit(x) # fit하고
x = scaler.transform(x) # 사용할 수 있게 바꿔서 저장하자
# print(x)

print("====================")
print(x_predict)
x_predict = scaler.transform(x_predict)
print(x_predict)
print("====================")



# 사이킷런의 model_selection에서 train_test_split을 불러온다
from sklearn.model_selection import train_test_split 
x_train,x_rest, y_train,y_rest = train_test_split(
    x, y, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x_test,x_val, y_test,y_val = train_test_split(
    x_rest,y_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔

print("x_train x.shape:", x_train.shape)

print("y_train.shape", y_train.shape)
print("y_val.shape", y_val.shape)
print("y_test.shape", y_test.shape)



# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape=(x.shape[1],1))
lstm = LSTM(75, activation='relu')(input1)
dense = Dense(180, activation='relu')(lstm)
dense = Dense(150, activation='relu')(dense)
dense = Dense(110, activation='relu')(dense)
dense = Dense(60, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output1 = Dense(1)(dense)
model = Model(inputs=input1, outputs=output1)

# model.summary()



# 3. 컴파일, 훈련
model.compile( # 컴파일
    loss='mse', # 오차함수는 mean squared error를 사용한다
    optimizer='adam',
    metrics=['mae']) # 최적화 방법은 'adam'을 사용한다

from tensorflow.keras.callbacks import TensorBoard # 텐서보드 = 웹페이지 시각화
from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=100,
    mode='auto',
    verbose=2)

to_hist = TensorBoard(
    log_dir='graph',
    histogram_freq=0,
    write_graph=True,
    write_images=True)

history = model.fit(x_train, y_train, #훈련, 일단 x_train, y_train 입력하고
    epochs=800, # 훈련 횟수
    verbose=0,
    validation_data=(x_val,y_val),
    callbacks=[early_stopping, to_hist]) # 0=로그 출력하지 않기, 1=막대그래프, 2=손실 정보

# 위와 같이 tensorboard를 추가하고,
# cmd창에서, graph폴더에 들어간 후, tensorboard --logdir=. 를 실행하면
# http://localhost:6006/ 주소에서 결과를 확인할 수 있다


# 4. 평가, 예측
# 평가 데이터 넣어서 결과 보기
loss, mae = model.evaluate(x_test, y_test)
print("loss: ", loss) # 이건 기본으로 나오고
print("mse: ", mae)




y_predict = model.predict(x_predict) # 평가 데이터 다시 넣어 예측값 만들기
# print("y_predict:\n", y)
print("y_predict:\n", y_predict)



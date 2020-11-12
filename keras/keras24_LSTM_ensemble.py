import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

from numpy import array

# 1.데이터
x1 = array([
    [1,2,3],[2,3,4],[3,4,5],[4,5,6],
    [5,6,7],[6,7,8],[7,8,9],[8,9,10],
    [9,10,11],[10,11,12],
    [20,30,40],[30,40,50],[40,50,60]
    ])
x2 = array([
    [10,20,30],[20,30,40],[30,40,50],[40,50,60],
    [50,60,70],[60,70,80],[70,80,90],[80,90,100],
    [90,100,110],[100,110,120],
    [2,3,4],[3,4,5],[4,5,6]
    ])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print("before x1.shape",x1.shape)
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) # (13,3,1)
print("after x1.shape",x1.shape)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) # (13,3,1)



# x1_predict = array([55,65,75])
# x2_predict = array([65,75,85])
# x1_predict = array([1,2,3])
# x2_predict = array([10,20,30])
x1_predict = x1 # array([40,50,60])
x2_predict = x2 # array([4,5,6])
x1_predict = x1_predict.reshape(13,3,1)
x2_predict = x2_predict.reshape(13,3,1)

### 실습 : 앙상블 함수형 모델을 만드시오




# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape=(3,1))
lstm1 = LSTM(64, activation='relu')(input1)
dense1 = Dense(512, activation='relu')(lstm1)
output1 = Dense(256)(dense1)

input2 = Input(shape=(3,1))
lstm2 = LSTM(64, activation='relu')(input2)
dense2 = Dense(512, activation='relu')(lstm2)
output2 = Dense(256)(dense2)

from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])
middle1 = Dense(128, name='middle1')(merge1)
middle1 = Dense(64, name='middle2')(middle1)
middle1 = Dense(32, name='middle3')(middle1)

output1_1 = Dense(16, name='output1_1')(middle1)
output1_1 = Dense(8, name='output1_2')(output1_1)
output1_1 = Dense(1, name='output1_3')(output1_1)


# 모델 정의
model = Model(inputs=[input1,input2], 
                outputs=output1_1)
model.summary()




# 3. 컴파일, 훈련
model.compile( # 컴파일
    loss='mse', # 오차함수는 mean squared error를 사용한다
    optimizer='adam') # 최적화 방법은 'adam'을 사용한다

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=100, 
    mode='auto', 
    verbose=2 )
hist = model.fit([x1,x2], y, 
    epochs=1000, # 훈련 횟수
    batch_size=1, # 훈련 데이터단위
    verbose=0,
    callbacks=[early_stopping]) # 훈련하고 호출하기, 리스트인걸 봐선 여러개 가능할 듯





# 4. 평가, 예측
# 평가 데이터 넣어서 결과 보기
result = model.evaluate([x1,x2], y, batch_size=1) 
print("result : ", result)


y_predict = model.predict([x1_predict, x2_predict])
print("y_predict:\n", y_predict)








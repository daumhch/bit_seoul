import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

# 인풋 2개, 아웃풋 1개 모델링 해보자

# 1.데이터
import numpy as np
x1 = np.array([range(1,101),range(711,811), range(100)])
x2 = np.array([range(4,104),range(761,861), range(100)])

y1 = np.array([range(101,201), range(311,411), range(100)])

x1 = x1.T
x2 = x2.T
print(x1.shape)
print(x2.shape)

y1 = y1.T


from sklearn.model_selection import train_test_split 

x1_train, x1_rest, y1_train, y1_rest = train_test_split(
    x1, y1, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x1_test, x1_val, y1_test, y1_val = train_test_split(
    x1_rest, y1_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔

# 3개의 데이터 까지 나눌 수 있다
x2_train, x2_rest, = train_test_split(
    x2, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x2_test, x2_val, = train_test_split(
    x2_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔


print("y1_train.size", y1_train.size)
print("y1_val.size", y1_val.size)
print("y1_test.size", y1_test.size)




# 2. 모델구성
from tensorflow.keras.models import Model # 함수형 모델 사용
from tensorflow.keras.layers import Dense, Input

# 모델 1
input1 = Input(shape=(3,))
dense1_1 = Dense(256, activation='relu', name='dense1_1')(input1)
dense1_2 = Dense(512, activation='relu', name='dense1_2')(dense1_1)
dense1_3 = Dense(1024, activation='relu', name='dense1_3')(dense1_2)
output1 = Dense(2048, name='output1')(dense1_3)
# model1 = Model(inputs=input1, outputs=output1)
# model1.summary()

# 모델 2
input2 = Input(shape=(3,))
dense2_1 = Dense(256, activation='relu', name='dense2_1')(input2)
dense2_2 = Dense(512, activation='relu', name='dense2_2')(dense2_1)
dense2_3 = Dense(1024, activation='relu', name='dense2_3')(dense2_2)
output2 = Dense(2048, name='output2')(dense2_3)
# model2 = Model(inputs=input2, outputs=output2)
# model2.summary()

# 모델 병합
# 대문자 소문자 본질적으로 같은데, 사용 방법이 다르다
from tensorflow.keras.layers import Concatenate, concatenate

# merge1 = concatenate([output1, output2]) # concatenate layer
# merge1 = Concatenate()([output1, output2])
merge1 = Concatenate(axis=1)([output1, output2])
middle1 = Dense(2048, name='middle1')(merge1)
middle2 = Dense(1024, name='middle2')(middle1)
middle3 = Dense(512, name='middle3')(middle2)

# 이렇게 해도 된다
# middle3 = Dense(30)(merge1)
# middle3 = Dense(7)(middle3)
# middle3 = Dense(11)(middle3)

# output 모델 구성(분기)
# middle layer 하기 싫으면 merge1을 바로 입력하면 된다
output1_1 = Dense(256, name='output1_1')(middle3)
output1_2 = Dense(128, name='output1_2')(output1_1)
output1_3 = Dense(3, name='output1_3')(output1_2)


# 모델 정의
model = Model(inputs=[input1,input2], 
                outputs=output1_3)
model.summary()



# 3. 컴파일, 훈련
model.compile( # 컴파일
    loss='mse', # 오차함수는 mean squared error를 사용한다
    optimizer='adam', # 최적화 방법은 'adam'을 사용한다
    metrics=['mae']) # mean absolute error 지표를 추가한다

#훈련, 일단 x_train, y_train 입력하고
model.fit([x1_train, x2_train], y1_train, 
    epochs=256, # 훈련 횟수
    batch_size=32, # 훈련 데이터단위
    validation_data=([x1_val, x2_val], y1_val), # 검증 데이터 사용하기
    verbose=0)

# 4. 평가, 예측
# 평가 데이터 넣어서 결과 보기
result = model.evaluate([x1_test,x2_test], y1_test, batch_size=32) 
print("result : ", result)

# output이 3개 -> loss 3개, mse3개가 출력되고, 전체 loss 포함해서 7개 출력
# output 별로 loss와 mse를 보고, 성능 개선 이유를 찾을 수 있다
# 전체 loss = output들의 loss 합


y1_predict = model.predict([x1_test,x2_test]) # 평가 데이터 다시 넣어 예측값 만들기
# print("y_predict:\n", y_predict)



# 사용자정의 RMSE 함수
# 사이킷런의 metrics에서 mean_squared_error을 불러온다
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y1_test, y1_predict))



# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y1_predict)
print("R2:", r2)


# 고정 값 커스텀 대입
x1_demo = np.full((1,3),[1,711,0])
x2_demo = np.full((1,3),[4,761,0])
y1_val_predict = model.predict([x1_demo,x2_demo]) # 평가 데이터 다시 넣어 예측값 만들기
print("y1[0]:\n", y1[0])
print("y1_val_predict:\n", y1_val_predict)





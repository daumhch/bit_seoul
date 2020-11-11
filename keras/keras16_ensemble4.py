# 인풋 1개, 아웃풋 3개 모델링 해보자

# 1.데이터
import numpy as np
x1 = np.array([range(1,101),range(711,811), range(100)])

y1 = np.array([range(101,201), range(311,411), range(100)])
y2 = np.array([range(501,601), range(431,531), range(100,200)])
y3 = np.array([range(501,601), range(431,531), range(100,200)])

x1 = x1.T
print(x1.shape)

y1 = y1.T
y2 = y2.T
y3 = y3.T

from sklearn.model_selection import train_test_split 

x1_train, x1_rest, y1_train, y1_rest = train_test_split(
    x1, y1, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x1_test, x1_val, y1_test, y1_val = train_test_split(
    x1_rest, y1_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔

# 3개의 데이터 까지 나눌 수 있다
y2_train, y2_rest, y3_train, y3_rest = train_test_split(
    y2, y3, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
y2_test, y2_val, y3_test, y3_val = train_test_split(
    y2_rest, y3_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔




# 2. 모델구성
from tensorflow.keras.models import Model # 함수형 모델 사용
from tensorflow.keras.layers import Dense, Input

# 모델 1
input1 = Input(shape=(3,))
dense1_1 = Dense(20, activation='relu', name='dense1_1')(input1)
dense1_2 = Dense(10, activation='relu', name='dense1_2')(dense1_1)
dense1_3 = Dense(5, activation='relu', name='dense1_3')(dense1_2)
output1 = Dense(3, name='output1')(dense1_3)
# model1 = Model(inputs=input1, outputs=output1)
# model1.summary()


# output 모델 구성(분기)
# middle layer 하기 싫으면 merge1을 바로 입력하면 된다
output1_1 = Dense(30, name='output1_1')(output1)
output1_2 = Dense(7, name='output1_2')(output1_1)
output1_3 = Dense(3, name='output1_3')(output1_2)

output2_0 = Dense(15, name='output2_0')(output1)
output2_1 = Dense(17, name='output2_1')(output2_0)
output2_3 = Dense(11, name='output2_3')(output2_1)
output2_4 = Dense(3, name='output2_4')(output2_3)

output3_0 = Dense(15, name='output3_0')(output1)
output3_1 = Dense(17, name='output3_1')(output3_0)
output3_3 = Dense(11, name='output3_3')(output3_1)
output3_4 = Dense(3, name='output3_4')(output3_3)

# 모델 정의
model = Model(inputs=input1, 
                outputs=[output1_3,output2_4,output3_4])
model.summary()



# 3. 컴파일, 훈련
model.compile( # 컴파일
    loss='mse', # 오차함수는 mean squared error를 사용한다
    optimizer='adam', # 최적화 방법은 'adam'을 사용한다
    metrics=['mae']) # mean absolute error 지표를 추가한다

#훈련, 일단 x_train, y_train 입력하고
model.fit(x1_train, [y1_train, y2_train, y3_train], 
    epochs=256, # 훈련 횟수
    batch_size=32, # 훈련 데이터단위
    validation_data=(x1_val, [y1_val, y2_val, y3_val]), # 검증 데이터 사용하기
    verbose=0)

# 4. 평가, 예측
# 평가 데이터 넣어서 결과 보기
result = model.evaluate(x1_test, [y1_test,y2_test, y3_test], batch_size=32) 
print("result : ", result)

# output이 3개 -> loss 3개, mse3개가 출력되고, 전체 loss 포함해서 7개 출력
# output 별로 loss와 mse를 보고, 성능 개선 이유를 찾을 수 있다
# 전체 loss = output들의 loss 합


y1_predict, y2_predict, y3_predict = model.predict(x1_test) # 평가 데이터 다시 넣어 예측값 만들기
# print("y_predict:\n", y_predict)


# 사용자정의 RMSE 함수
# 사이킷런의 metrics에서 mean_squared_error을 불러온다
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE( y1_test, y1_predict))
print("RMSE:", RMSE( y2_test, y2_predict))
print("RMSE:", RMSE( y3_test, y3_predict))


# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2_3 = r2_score(y3_test, y3_predict)
print("R2:", r2_1)
print("R2:", r2_2)
print("R2:", r2_3)


# 다차원 행렬 들어가기 전 까지 모든 파트를 반영한 소스

import timeit
start_time = timeit.default_timer() # 시작 시간 체크

# 1.데이터
import numpy as np # numpy를 불러오고 앞으로 이름을 np로 쓴다
x = np.array(range(1, 101)) # 1부터 101-1까지 숫자를 array로 만든다
y = np.array(range(101, 201))

# 사이킷런의 model_selection에서 train_test_split을 불러온다
from sklearn.model_selection import train_test_split 

# 데이터는 train / val / test로 나뉘고 비율은 6:2:2를 보통 쓴다
# train_test_split를 사용하면 랜덤하게 뽑아서 나눠준다
x_train, x_rest, y_train, y_rest = train_test_split(
    x, y, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x_test, x_val, y_test, y_val = train_test_split(
    x_rest, y_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔



# 2. 모델
from tensorflow.keras.models import Sequential # 순차적인분석
from tensorflow.keras.layers import Dense # Dense layer를 사용

model = Sequential() # Sequential 클래스 불러와서 객체 생성
# 입력은 1차원, 곧바로 256개 node, 입력레이어+히든레이어1
model.add(Dense(256, input_dim=1)) 
model.add(Dense(512)) # 히든레이어2, node 512개
model.add(Dense(256)) # 히든레이어3, node 256개
model.add(Dense(1)) # 출력레이어



# 3. 컴파일, 훈련
model.compile( # 컴파일
    loss='mse', # 오차함수는 mean squared error를 사용한다
    optimizer='adam', # 최적화 방법은 'adam'을 사용한다
    metrics=['mae']) # mean absolute error 지표를 추가한다

model.fit(x_train, y_train, #훈련, 일단 x_train, y_train 입력하고
    epochs=256, # 훈련 횟수
    batch_size=32, # 훈련 데이터단위
    validation_data=(x_val, y_val), # 검증 데이터 사용하기
    verbose=0) # 0=로그 출력하지 않기, 1=막대그래프, 2=손실 정보



# 4. 평가, 예측
# 평가 데이터 넣어서 결과 보기
loss, mae = model.evaluate(x_test, y_test, batch_size=32) 
print("loss : ", loss) # 이건 기본으로 나오고
print("mae : ", mae) # 이건 metrics에 추가한 것

y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
# print("y_predict:\n", y_predict)



# 사용자정의 RMSE 함수
# 사이킷런의 metrics에서 mean_squared_error을 불러온다
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_test, y_predict))



# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2:", r2)

# print("x_train.size", x_train.size)
# print("x_val.size", x_val.size)
# print("x_test.size", x_test.size)


terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time)) 



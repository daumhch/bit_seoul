# train_test_splite을 사용하지 않고 변환하기 = 슬라이싱
# multi layer perceptron

# 1.데이터
import numpy as np
x = np.array([range(1,101),range(711,811), range(100)])
y = np.array([range(101,201), range(311,411), range(100)])

# x = 100행 3열, 데이터 종류 3가지
# print(x)
print("before x.shape:", x.shape) # 출력: (3,) array에 range만 3개 저장되어 있다는 뜻

# (100, 3)의 형태로 만들어야 한다

# x = x.reshape(3,100).T
# x = np.transpose(x)
# x = x.transpose()
x = x.T
print("x.shape:", x.shape)

y = y.T
# print(y.shape)
# print("after y.shape:", y.shape)

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

# print("x_train.size", x_train.size)
# print("x_val.size", x_val.size)
# print("x_test.size", x_test.size)


y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]




# 구성하고자 하는 모델 : y1, y2, y3 = w1x1 + w2x2 + w3x3 + b

# 2. 모델구성
from tensorflow.keras.models import Sequential # 순차적인분석
from tensorflow.keras.layers import Dense # Dense layer를 사용

model = Sequential() # Sequential 클래스 불러와서 객체 생성
model.add(Dense(256, input_dim=3)) 
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(3)) # 입력이 3개니까 출력도 3개



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







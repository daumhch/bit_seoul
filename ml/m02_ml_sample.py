import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

# 1.데이터
# 1.1 load_data
# 1.2 train_test_split
# 1.3 scaler
# 1.4 reshape
# 2.모델
# 3.컴파일 훈련
# 4.평가 예측



# 머신러닝 로드하기
from sklearn.svm import LinearSVC


# 1.데이터
# 1.1 load_data
import numpy as np
from sklearn.datasets import load_iris
datasets = load_iris()
x = datasets.data
y = datasets.target
print("========== 데이터 로딩 끝 ==========")

# LinearSVC는 OneHotEncoding이 필요없다
# OneHotEncoding
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2, random_state=44)
print("========== train_test_split 끝 ==========")

# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print("========== scaler 끝 ==========")

# 1.4 reshape
# x가 2차원이라, 별도의 reshape 없이 그대로 x를 사용한다
print("========== reshape 끝 ==========")






# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# model = Sequential()
# model.add(Dense(50, activation='relu', input_shape=(x_train.shape[1],) ))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(3, activation='softmax'))
# model.summary()

# LinearSVC
model = LinearSVC()



# 3. 컴파일, 훈련
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy'])

# history = model.fit(x_train, y_train,
#     epochs=500,
#     verbose=1,
#     validation_split=0.25,
#     batch_size=512)

# LinearSVC
model.fit(x_train, y_train)


# 4. 평가 및 예측
# loss, accuracy = model.evaluate(x_test, y_test, batch_size=512)
# print("loss: ", loss)
# print("accuracy: ", accuracy)

# LinearSVC
score = model.score(x_test, y_test)
print("score: ", score)


'''
# 평가 데이터 다시 넣어 예측값 만들기
y_predict = model.predict(x_test) 
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


'''




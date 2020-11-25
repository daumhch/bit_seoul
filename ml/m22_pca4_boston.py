# PCA로 축소해서 DNN 모델을 완성하시오
# 1. 0.95이상
# 2. 1이상
# mnist DNN과 loss와 acc를 비교하시오


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


# 1.데이터
# 1.1 load_data
import numpy as np
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data
y = datasets.target

print("x.shape:", x.shape) # (70000, 28, 28)

# OneHotEncoding
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

# 1.4 reshape
# # PCA를 위해서 reshape
# x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
print("after reshape x.shape:", x.shape) # (70000, 28, 28)




# 1.5 PCA
from sklearn.decomposition import PCA
cumsum_standard = 0.95
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= cumsum_standard) +1
print("n_components:",d) # 202

pca = PCA(n_components=d)
x = pca.fit_transform(x)
print("after pca x.shape", x.shape) # (70000, 202)





# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2)
print("split x_train.shape:",x_train.shape, x_test.shape)



# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],) ))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
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
    validation_split=0.25,
    callbacks=[early_stopping],
    batch_size=128)



# 4. 평가 및 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=128)
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



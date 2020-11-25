# PCA로 축소해서 DNN 모델을 완성하시오
# 1. 0.95이상
# 2. 1이상
# mnist DNN과 loss와 acc를 비교하시오


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


# 1.데이터
# 1.1 load_data
import numpy as np
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print("x.shape:", x.shape) # (70000, 28, 28)

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

# 1.4 reshape
# # PCA를 위해서 reshape
# x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
print("after reshape x.shape:", x.shape) # (70000, 28, 28)


# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)


# 1.5 PCA
from sklearn.decomposition import PCA
cumsum_standard = 0.999
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= cumsum_standard) +1
print("n_components:",d)


pca = PCA(n_components=d)
x = pca.fit_transform(x)
print("after pca x.shape", x.shape)





# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2)
print("split x_train.shape:",x_train.shape, x_test.shape)



# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(x_train.shape[1],) ))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.summary()



# 3. 컴파일, 훈련
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='accuracy',
    patience=50,
    mode='auto',
    verbose=2)

history = model.fit(x_train, y_train,
    epochs=500,
    verbose=0,
    validation_split=0.2,
    callbacks=[early_stopping], 
    batch_size=128)




# 4. 평가 및 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("accuracy: ", accuracy)


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




import matplotlib.pyplot as plt
#모델 시각
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
acc_ax.plot(history.history['accuracy'], 'b', label='train accuracy')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val accuracy')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax. set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
 
plt.show()
model.summary()



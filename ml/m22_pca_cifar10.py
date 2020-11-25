# PCA로 축소해서 DNN 모델을 완성하시오
# 1. 0.95이상
# 2. 1이상
# mnist DNN과 loss와 acc를 비교하시오


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


# 1.데이터
# 1.1 load_data
import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
print("x.shape:", x.shape) # (70000, 28, 28)

x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
print("after reshape x.shape:", x.shape) # (70000, 28, 28)


# 1.5 PCA
from sklearn.decomposition import PCA
cumsum_standard = 1
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= cumsum_standard) +1
print("n_components:",d) # 154

pca = PCA(n_components=d)
x = pca.fit_transform(x)
print("after pca x.shape", x.shape) # (70000, 154)




# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)



# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, train_size=50000, test_size=10000)
print("split shape:",x_train.shape, x_test.shape)



# 1.4 reshape
# PCA를 위해서 위에서 했음




# 이미지를 DNN 할 때에는 reshape하고 scaler 하자
# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) # fit하고
x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자
x_test = scaler.transform(x_test) # 사용할 수 있게 바꿔서 저장하자


print("x_train.shape:", x_train.shape)





# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],) ))
model.add(Dense(20, activation='relu') )
model.add(Dense(30, activation='relu') )
model.add(Dense(20, activation='relu') )
model.add(Dense(10, activation='relu') )
model.add(Dense(10, activation='relu') )
model.add(Dense(10, activation='softmax') )
model.summary()




# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy', #CNN은 웬만하면 categorical_crossentropy
    optimizer='adam',
    metrics=['accuracy'] # 이젠 accuracy를 관찰할만하다
    )

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=10,
    mode='auto',
    verbose=2)

history = model.fit(
    x_train, y_train,
    
    # 수치 계산이 아닌, 데이터 7만개 훈련이니, epoch이 그리 크지 않아도 되는 것...맞는지 확인해보자
    epochs=100, 
    batch_size=512,
    verbose=0, # 0=로그 출력하지 않기, 1=막대그래프, 2=손실 정보
    
    # 별도의 validation 데이터를 split 하지 않았으니, train에서 잘라 쓴다
    validation_split=0.2, 
    
    callbacks=[early_stopping]) 



# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=512)
print("loss: ", loss)
print("accuracy: ", accuracy)


y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
# print("y_predict:\n", y)
print("y_test:\n", y_test)
print("y_predict:\n", y_predict)


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])

plt.title('loss & accuracy')
plt.ylabel('loss, accuracy')
plt.xlabel('epoch')

plt.legend(['loss', 'accuracy']) # 색인
plt.show()


print("cifar10 dnn end")







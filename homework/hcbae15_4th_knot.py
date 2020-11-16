import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# CNN = Convolution Neural Network
# 주로 이미지, 그러나 마냥 이미지만 하는 것 아니고,
# DNN으로도 가능하긴 하다

# 대략 원리
# 이미지는 기본 3차원이다
# LSTM에서, batch_size, timesteps, feature(몇개씩 자를건지) 로 구분짓고,
# CNN에서는, batch_size, row, column, channels(흑백:1, 칼라:3)으로 구분한다

# batch_size는 '행무시'에 의해 고려하지 않는다(데이터 사이즈 일 뿐)
# 따라서 CNN의 input_shape는 3차원으로, (rows, columns, channels) 이다

# model에서 Conv2D 입력 파라미터는,
# filters, kernel_size, strides, padding, input_shape, activation 등이 있다

# filters는 노드 갯수
# kernel_size는 자를 조각 이미지 크기 일반적으로 (2,2)
# strides는 자르고 이동하는 단계, 1이면 1칸씩, default=1
# padding은 입력과 출력을 같게(same)하거나 축소(valid)하는 설정
# input_shape은 입력되는 데이터 모양, (row, column, channels)이다
# activation은 활성화함수, CNN마지막은 꼭 activation='softmax'이다

# CNN other Layers
# MaxPooling2D()는, 
# pool_size는 축소할 공간 크기
# strides는 축소하고 이동하는 단계, 1이면 1칸씩

# Flatten()은,
# 2차원 데이터를 일렬로 만드는 레이어
# 일렬로 만들면 Dense에서 사용할 수 있다

# CNN의 output은,
# 결과값(y)의 카테고리 갯수만큼 존재한다
# 카테고리가 10종류라면, Dense(10)



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print("origin x shape:",x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
# print("origin y shape:",y_train.shape, y_test.shape) # (60000,) (10000,)
# print(x_train[0]) # 28x28 리스트 데이터
# print(y_train[0]) # 5
print("y_train[1]:",y_train[1]) # 5




# OneHotEncoding = 결과값의 카테고리화 = 종류별로 특이값을 갖도록 = 동일 가중치
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # (60000,10) (10000,10)
# print(y_train[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] = 5
print("categorical y_train[1]:",y_train[1]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] = 5




# CNN을 위한 reshape = 3차원을 4차원으로
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
# shape 뒷부분이 28, 28 일 필요는 없지만, 일단 그대로 가져다 사용한다
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print("reshape x:", x_train.shape, x_test.shape)




# Scaler
# 선택은 아무거나, 최적이라 생각하는 주관적 판단
# MaxMinScaler는 2차원만 된다
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
# print(x_train[0])



# 10000개 데이터 중에 99980개는 validation, 20개는 test
# train_size=0.002, test_size=0.998
from sklearn.model_selection import train_test_split 
x_test,x_val, y_test,y_val = train_test_split(
    x_test,y_test, train_size=0.1, test_size=0.9)

print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)
print("after x_val.shape",x_val.shape)




# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add( Conv2D(32, (2,2), padding='same', input_shape=(28,28,1) ) )
model.add(MaxPooling2D(pool_size=(2,2) ))
model.add( Conv2D(64, (2,2) ) )
model.add(MaxPooling2D(pool_size=(2,2) ))
model.add(Flatten())
model.add(Dense(20, activation='relu') )
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
    patience=100,
    mode='auto',
    verbose=2)

model.fit(
    x_train, y_train,
    
    # 수치 계산이 아닌, 데이터 7만개 훈련이니, epoch이 그리 크지 않아도 되는 것...맞는지 확인해보자
    epochs=15, 
    batch_size=128,
    verbose=1, # 0=로그 출력하지 않기, 1=막대그래프, 2=손실 정보
    
    # 별도의 validation 데이터를 split 하지 않았으니, train에서 잘라 쓴다
    validation_split=0.2, 
    
    callbacks=[early_stopping]) 



# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("accuracy: ", accuracy)


y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
# print("y_predict:\n", y)
print("y_test:\n", y_test)
print("y_predict:\n", y_predict)






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
# https://buomsoo-kim.github.io/keras/2018/05/05/Easy-deep-learning-with-Keras-11.md/
# 위 링크 보고 뭔가 개선해봄

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization

model = Sequential()
model.add( Conv2D(8, (3,3), padding='same', input_shape=(28,28,1)) )
model.add( Conv2D(8, (1,1), padding='same') )
model.add(MaxPooling2D())

model.add( Conv2D(16, (3,3), padding='same') )
model.add( Conv2D(16, (1,1), padding='same') )
model.add(MaxPooling2D())


# prior layer should be flattend to be connected to dense layers
model.add(Flatten())
# dense layer with 50 neurons
model.add(Dense(50, activation = 'relu'))
model.add(Dense(10, activation = 'softmax') )
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
    batch_size=128,
    verbose=0, # 0=로그 출력하지 않기, 1=막대그래프, 2=손실 정보
    
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



plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()








'''
여러 테스트 후 최적이라 생각하는 설정을 위와 같이 했다
적은 total params, 낮은 early stoppting, 낮은 loss, 높은 accuracy를 추구했다

patience 10 epoch 100


Network In Network (NIN)
1-D conv

(2,2)->(2,2) 4 단위 층
Total params: 37,310
Epoch 00071: early stopping
loss:  0.0940813273191452
accuracy:  0.9800000190734863

(2,2)->(1,1) 4 단위 층
Total params: 22,310
Epoch 00097: early stopping
loss:  0.07866955548524857
accuracy:  0.9860000014305115

(3,3)->(1,1) 4 단위 층
Total params: 41,310

1st train
Epoch 00039: early stopping
loss:  0.033764030784368515
accuracy:  0.9909999966621399

2nd train
Epoch 00059: early stopping
loss:  0.06788752973079681
accuracy:  0.984000027179718

(3,3)->(1,1)->batchNormal
Total params: 41,710
Trainable params: 41,510
Non-trainable params: 200
1st train
Epoch 00052: early stopping
loss:  0.049033768475055695
accuracy:  0.9919999837875366

2nd train
Epoch 00061: early stopping
loss:  0.04857822507619858
accuracy:  0.9909999966621399




VGGNet
16x2 -> 32x2 -> 64x2 -> 128x2
Total params: 300,192
Epoch 00039: early stopping
loss:  0.04993342235684395
accuracy:  0.9869999885559082

16x2 -> 32x2 -> 64x2
Total params: 101,152
Epoch 00031: early stopping
loss:  0.039570145308971405
accuracy:  0.9869999885559082

32x2 -> 64x2 -> 128x2
Total params: 344,592
끝까지
loss:  0.019834283739328384
accuracy:  0.996999979019165

64x2 -> 128x2
Total params: 573,168
Epoch 00034: early stopping
loss:  0.1277066320180893
accuracy:  0.9879999756813049

8x2+(1,1) -> 16x2+(1,1)
1st train
Total params: 44,256
Epoch 00038: early stopping
loss:  0.06576678156852722
accuracy:  0.9900000095367432

2nd train
Epoch 00054: early stopping
loss:  0.10196372121572495
accuracy:  0.9810000061988831
'''
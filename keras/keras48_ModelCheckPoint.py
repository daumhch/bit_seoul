import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# OneHotEncoding



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("origin x shape:",x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print("origin y shape:",y_train.shape, y_test.shape) # (60000,) (10000,)

# print(x_train[0]) # 28x28 리스트 데이터
# print(y_train[0]) # 5







# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (60000,10) (10000,10)
print(y_train[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]




# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
# shape 뒷부분이 28, 28 일 필요는 없지만, 일단 그대로 가져다 사용한다
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print("reshape x:", x_train.shape, x_test.shape)



# Scaler
# 선택은 아무거나, 최적이라 생각하는 주관적 판단
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
# print(x_train[0])





# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add( Conv2D(32, (2,2), padding='same', input_shape=(28,28,1) ) )
model.add(MaxPooling2D(pool_size=(2,2) ))
model.add( Conv2D(64, (2,2) ) )
model.add(MaxPooling2D(pool_size=(2,2) ))
model.add(Flatten())
model.add(Dense(28,activation='relu'))
model.add(Dense(28*2,activation='relu'))
model.add(Dense(28*3,activation='relu'))
model.add(Dense(10, activation='softmax') )
model.summary()



# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=2)

from tensorflow.keras.callbacks import ModelCheckpoint # 모델 체크 포인트
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
model_check_point = ModelCheckpoint(
    filepath=modelpath,
    monitor='val_loss',
    save_best_only=True,
    mode='auto')

hist = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=32,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping, model_check_point])



# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss: ", result[0])
print("accuracy: ", result[1])



# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) # 단위는 찾아보자

plt.subplot(2,1,1) # 2장 중에 첫 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2) # 2장 중에 두 번째
plt.plot(hist.history['accuracy'], marker='.', c='red')
plt.plot(hist.history['val_accuracy'], marker='.', c='blue')
plt.grid()
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['accuracy', 'val_accuracy'])

plt.show()




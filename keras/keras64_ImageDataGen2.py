# 넘파이 불러와서
# .fit 으로 코딩

# 남자 여자를
# 넘파이 저장
# fit_generator로 코딩

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(44)

x_train = np.load('./data/keras64_mf_train_x.npy')
y_train = np.load('./data/keras64_mf_train_y.npy')
x_test = np.load('./data/keras64_mf_test_x.npy')
y_test = np.load('./data/keras64_mf_test_y.npy')


print(x_train.shape)

# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense 
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu',
    input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
    ))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=1))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()


# 3. 컴파일, 훈련
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='auto',
    verbose=2)

hist = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# 4. 평가, 예측

scores = model.evaluate(x_test, y_test, batch_size=128)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
# accuracy


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



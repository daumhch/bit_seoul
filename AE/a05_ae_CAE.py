# a04_ae_mlp.py를 카피
# CNN 구성
# 인코더 부분을 CNN 2~3개로 구성
# padding = same 과 valid 비교하시오

import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data() # 이미지에 대한 라벨이 없는 경우

# CNN을 위한 reshape = 3차원을 4차원으로
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
# shape 뒷부분이 28, 28 일 필요는 없지만, 일단 그대로 가져다 사용한다
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print("reshape x:", x_train.shape, x_test.shape)

# x_train = x_train.reshape(60000, 784).astype('float32')/255.
# x_test = x_test.reshape(10000, 784)/255. # 윗 줄과 같은 표현
x_train = x_train.astype('float32')/255.
x_test = x_test/255. # 윗 줄과 같은 표현

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D
from tensorflow.keras.layers import Flatten

def autoencoder(hidden_layer_size):
    model = Sequential()
    # model.add(Dense(units=hidden_layer_size, input_shape=(784,),
    #                 activation='relu'))
    # model.add(Dense(units=512, activation='relu'))
    # model.add(Dense(units=256, activation='relu'))
    # model.add(Dense(units=128, activation='relu'))
    model.add(Conv2D(filters=hidden_layer_size, 
                    kernel_size=(3,3), 
                    padding='same',
                    input_shape=(28,28,1)))
    model.add(Conv2D(filters=256, 
                    kernel_size=(3,3), 
                    padding='same'))
    model.add(Conv2D(filters=128, 
                    kernel_size=(3,3), 
                    padding='same'))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154) # PCA 0.95 효과와 같은 컬럼 갯수

# model.compile(optimizer='adam', 
#                 loss='mse',
#                 metrics=['accuracy'])

model.compile(optimizer='adam', 
                loss='binary_crossentropy',
                metrics=['accuracy'])


model.fit(x_train, x_train.reshape(60000,784), epochs=5, batch_size=512)

output = model.predict(x_test)


from matplotlib import pyplot as plt
import random

fig, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10)) = plt.subplots(2,5,figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


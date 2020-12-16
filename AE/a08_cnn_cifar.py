import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = cifar10.load_data() # 이미지에 대한 라벨이 없는 경우


# x_train = x_train.reshape(60000, 784).astype('float32')/255.
# x_test = x_test.reshape(10000, 784)/255. # 윗 줄과 같은 표현
x_train = x_train.astype('float32')/255.
x_test = x_test/255. # 윗 줄과 같은 표현


x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

# 노이즈 추가하면 1이 넘는다 추가 스케일링이 필요하다
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)




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
                    input_shape=(32,32,3)))
    model.add(Conv2D(filters=128, 
                    kernel_size=(3,3), 
                    padding='same'))
    model.add(Conv2D(filters=64, 
                    kernel_size=(3,3), 
                    padding='same'))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=3072, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154) # PCA 0.95 효과와 같은 컬럼 갯수

# model.compile(optimizer='adam', 
#                 loss='mse',
#                 metrics=['accuracy'])

model.compile(optimizer='adam', 
                loss='binary_crossentropy',
                metrics=['accuracy'])


model.fit(x_train_noised, x_train.reshape(50000,3072), epochs=5, batch_size=512)

output = model.predict(x_test_noised)


from matplotlib import pyplot as plt
import random

fig, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10) , (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3,5,figsize=(15,7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(32,32,3), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(32,32,3), cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_images[i]].reshape(32,32,3), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


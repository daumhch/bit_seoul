import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data() # 이미지에 대한 라벨이 없는 경우

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784)/255. # 윗 줄과 같은 표현


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img) # 14->15 = PCA 기능
decoded = Dense(784, activation='sigmoid')(encoded) # 15->16 = 데이터 전처리(0~1사이로)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy') # 이것도 된다

autoencoder.fit(x_train, x_train, # y는 자기 자신
                epochs=50, 
                batch_size=256,
                validation_split=0.2)


decoded_imgs = autoencoder.predict(x_test)


import matplotlib.pyplot as plt
n=10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()






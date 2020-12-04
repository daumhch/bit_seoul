# 최적화 튠으로 구성하시오

# 가장 좋은 놈이 어떤 건지 결과치 비교용
# 기본튠 + 전이학습 9개 모델 비교

# 9개 전이학습 모델들은
# Flatten() 다음에는 모두 똑같은 레이어로 구성할 것



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import timeit

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("x_train.shape:", x_train.shape) # x_train.shape: (50000, 32, 32, 3)
print("x_test.shape:", x_test.shape) # x_test.shape: (10000, 32, 32, 3) 



# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)
print(y_train[0]) # [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]  



# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_train.shape[3])
print("reshape x:", x_train.shape, x_test.shape)


# Scaler
# 선택은 아무거나, 최적이라 생각하는 주관적 판단
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
# print(x_train[0])



start_time = timeit.default_timer() # 시작 시간 체크

# 2.모델
model = Sequential()
model.add( Conv2D(32, (3,3), padding='same', input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])) )
model.add( Conv2D(32, (1,1), padding='valid') )
model.add( Conv2D(32, (3,3), padding='same') )
model.add(MaxPooling2D(pool_size=(2,2)))

model.add( Conv2D(64, (3,3), padding='same') )
model.add( Conv2D(64, (1,1), padding='valid') )
model.add( Conv2D(64, (3,3), padding='same') )
model.add(MaxPooling2D(pool_size=(2,2)))

model.add( Conv2D(128, (3,3), padding='same') )
model.add( Conv2D(128, (1,1), padding='valid') )
model.add( Conv2D(128, (3,3), padding='same') )
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(10, activation = 'softmax') )
# model.summary()




# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy', #CNN은 웬만하면 categorical_crossentropy
    optimizer='adam',
    metrics=['accuracy'] # 이젠 accuracy를 관찰할만하다
    )

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='auto',
    verbose=0)

history = model.fit(
    x_train, y_train,
    epochs=30, 
    batch_size=512,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping]) 

end_time = timeit.default_timer() # 시작 시간 체크


# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=512)
print("loss: %3.9f" %loss, "\t /accuracy: %3.9f" %accuracy, "\t /time: ", (end_time - start_time))

# Original
# loss: 1.128875494        /accuracy: 0.609300017          /time:  65.75466949999999

# VGG16
# loss: 1.128554583        /accuracy: 0.610599995          /time:  70.1103181

# VGG19
# loss: 1.158567667        /accuracy: 0.598599970          /time:  81.472763

# Xception
# ValueError: Input size must be at least 71x71; got `input_shape=(32, 32, 3)`

# ResNet101
# loss: 1.893944740        /accuracy: 0.310000002          /time:  115.4373645

# ResNet50
# loss: 1.880036354        /accuracy: 0.325500011          /time:  133.6047665

# InceptionV3
# ValueError: Input size must be at least 75x75; got `input_shape=(32, 32, 3)`

# InceptionResNetV2
# ValueError: Input size must be at least 75x75; got `input_shape=(32, 32, 3)`

# DenseNet121
# loss: 1.043872714        /accuracy: 0.647199988          /time:  49.207367

# MobileNetV2
# loss: 1.800943136        /accuracy: 0.349700004          /time:  59.7683174

# NASNetMobile
# ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).






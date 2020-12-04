import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

import timeit


# 데이터 준비
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



## OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Scaler
# 선택은 아무거나, 최적이라 생각하는 주관적 판단
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


start_time = timeit.default_timer()

# 모델
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

head_model = ResNet152V2(include_top=False, 
                input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]) ) # 14,714,688
head_model.trainable = False

model = Sequential()
model.add(head_model)
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(10, activation='softmax'))
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

end_time = timeit.default_timer()


# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=512)
print("loss: %3.9f" %loss, "\t /accuracy: %3.9f" %accuracy, "\t /time: ", (end_time - start_time))

# loss: 1.880036354        /accuracy: 0.325500011          /time:  133.6047665

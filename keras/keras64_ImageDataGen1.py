# 남자 여자를
# 넘파이 저장
# fit_generator로 코딩

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(44)

train_datagen = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2,
    )

mf_train = train_datagen.flow_from_directory(
    './data/data2',
    target_size=(150,150),
    batch_size=5, # 200은 에러난다, 총 이미지가 160이니까
    class_mode='binary',
    subset='training'
    # , save_to_dir='./data/data1_2/train'
    )
mf_test = train_datagen.flow_from_directory(
    './data/data2',
    target_size=(150,150),
    batch_size=5, # 200은 에러난다, 총 이미지가 160이니까
    class_mode='binary',
    subset='validation'
    # , save_to_dir='./data/data1_2/train'
    )

print(type(mf_train)) 
print(mf_train[0][0].shape)

print("mf_train len:",len(mf_train))
print("mf_test len:",len(mf_test))

# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense 
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu',
    input_shape=(mf_train[0][0].shape[1],mf_train[0][0].shape[2],mf_train[0][0].shape[3])
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

hist = model.fit_generator(
    mf_train,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = mf_test,
    validation_steps = 4,
    callbacks=[early_stopping]
)

# 4. 평가, 예측

scores = model.evaluate_generator(mf_test, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
# accuracy



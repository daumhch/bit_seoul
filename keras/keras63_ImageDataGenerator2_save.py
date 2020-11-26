import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


# 폴더를 가져와서 데이터화 해줌
# 폴더에 들어가는 이미지 크기가 일정해야 함

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(44)

# 이미지에 대한 생성 옵션 정하기
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
    )

test_datagen = ImageDataGenerator(rescale=1./255.)

# flow 또는 flow_from_directory
# 실제 데이터가 있는 곳을 알려주고, 이미지를 불러오는 작업

xy_train = train_datagen.flow_from_directory(
    './data/data1/train',
    target_size=(150,150),
    batch_size=5, # 200은 에러난다, 총 이미지가 160이니까
    class_mode='binary'
    # , save_to_dir='./data/data1_2/train'
    )

xy_test = test_datagen.flow_from_directory(
    './data/data1/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
    )

print(type(xy_train)) 
# <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>

# batch_size를 5로 놓았을 때
# # print(xy_train.shape) # error
# # print(xy_train[0].shape) # error
# print(xy_train[0][0].shape) # (5, 150, 150, 3)
# print(xy_train[0][1].shape) # (5,)
# # print(xy_train[0][2].shape) # error
# print(xy_train[1][0].shape) # (5, 150, 150, 3)
# print(xy_train[1][1].shape) # (5,)
# print(xy_train[2][0].shape) # (5, 150, 150, 3)
# print(xy_train[2][1].shape) # (5,)

# print(len(xy_train)) # 32, 160장 / batch_size 5 = 32 len

# print(xy_train[0][0][0]) # 첫 번째 이미지
# print(xy_train[0][0][0].shape) # (150, 150, 3)

# print(xy_train[0][1][:10]) # [0. 0. 1. 1. 0.] # batch_size 때문에 5개
print("========== 데이터 로딩 끝 ==========")

'''
print("========== numpy save 시작 ==========")
np.save('./data/keras63_train_x.npy', arr=xy_train[0][0])
np.save('./data/keras63_train_y.npy', arr=xy_train[0][1])
np.save('./data/keras63_test_x.npy', arr=xy_test[0][0])
np.save('./data/keras63_test_y.npy', arr=xy_test[0][1])
print("========== numpy save 끝 ==========")
'''

print(xy_train[0][0].shape)

# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense 
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu',
    input_shape=(xy_train[0][0].shape[1],xy_train[0][0].shape[2],xy_train[0][0].shape[3])
    ))
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

hist = model.fit_generator(
    xy_train,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = xy_test,
    validation_steps = 4
)

# 4. 평가, 예측
# result = model.evaluate(xy_train, xy_test, batch_size=5)
# print("loss: ", result[0])
# print("accuracy: ", result[1])

scores = model.evaluate_generator(xy_test, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
# accuracy



# 시각화




import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# 전이학습 = 다른 사람이 만든거, 가중치까지 빼서 가져다 쓴다

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.models import Sequential

# model = VGG16() # parameter = 138,357,544
# model = VGG16(weights='imagenet') # parameter = 138,357,544

# input_shape를 바꾸려면, include_tope을 false로 놓고, input_shape를 입력한다
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3) ) # 14,714,688

# 새로 훈련시킬꺼야
# model.trainable = True # Trainable params: 14,714,688

# 훈련 시키지 않고, imagenet train을 사용할꺼야
vgg16.trainable = False # Non-trainable params: 14,714,688

vgg16.summary()

print('동결하기 전 훈련되는 가중치의 수')
print(':', len(vgg16.trainable_weights)) 
# VGG16() 모델 그대로 만들면 32 = 16개의 w, 16개의 b
# trainable = False 라면, 가중치 0


# 모델 출력부분 붙이기
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
# model.add(BatchNormalization()) # 가중치 2 증가한다, 가중치 연산을 한다
# model.add(Dropout(0.2)) # 가중치가 늘어나지 않는다
model.add(Activation('relu')) # 가중치가 늘어나지 않는다
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))
model.summary()

print('모델 출력 붙이고 난 후 가중치의 수')
print(':', len(model.trainable_weights)) 


import pandas as pd
pd.set_option('max_colwidth',None)
layers = [(layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns=['Layer Name', 'Layer Trainable'])
print(aaa)



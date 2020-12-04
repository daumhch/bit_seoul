import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# 추가구현
# VGG19, Xception
# ResNet 시리즈들
from tensorflow.keras.applications import VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2
from tensorflow.keras.applications import ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.models import Sequential

vgg16 = VGG16()
vgg16.trainable = True
# vgg16.summary()
print('weight:', len(vgg16.trainable_weights)) 
# VGG16() 모델 그대로 만들면 32 = 16개의 w, 16개의 b
# trainable = False 라면, 가중치 0


# defalut 모델의 파라미터 갯수와 가중치 수를 정리하시오
# ex)
# VGG 16 / 138,357,544 / 32

model_list = [
    VGG16, VGG19, Xception,
    ResNet101, ResNet101V2,
    ResNet152, ResNet152V2,
    ResNet50, ResNet50V2,
    InceptionResNetV2, InceptionV3,
    MobileNet, MobileNetV2,
    DenseNet121, DenseNet169, DenseNet201,
    NASNetLarge, NASNetMobile
]

for cnt in range(len(model_list)):
    model = model_list[cnt]()
    model.trainable = True
    print('name: %20s' %model.name,"\t/ %10d" %model.count_params(),"\t/", len(vgg16.trainable_weights)) 

# name:                vgg16      /  138357544    / 32
# name:                vgg19      /  143667240    / 32
# name:             xception      /   22910480    / 32
# name:            resnet101      /   44707176    / 32
# name:          resnet101v2      /   44675560    / 32
# name:            resnet152      /   60419944    / 32
# name:          resnet152v2      /   60380648    / 32
# name:             resnet50      /   25636712    / 32
# name:           resnet50v2      /   25613800    / 32
# name:  inception_resnet_v2      /   55873736    / 32
# name:         inception_v3      /   23851784    / 32
# name:   mobilenet_1.00_224      /    4253864    / 32
# name: mobilenetv2_1.00_224      /    3538984    / 32
# name:          densenet121      /    8062504    / 32
# name:          densenet169      /   14307880    / 32
# name:          densenet201      /   20242984    / 32
# name:               NASNet      /   88949818    / 32
# name:               NASNet      /    5326716    / 32


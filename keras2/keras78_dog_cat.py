import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# 데이터 준비
# from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

img_dog = load_img('./data/dog_cat/개1.jpg', target_size=(224,224))
img_cat = load_img('./data/dog_cat/고양이1.jpg', target_size=(224,224))
img_ap = load_img('./data/dog_cat/ap.jfif', target_size=(224,224))


# plt.imshow(img_dog)
# plt.show()

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_ap = img_to_array(img_ap)

print(arr_dog)
print(type(arr_dog))
print(arr_dog.shape) # (224, 224, 3)


# keras로 가져올 때는 RGB를 BGR로 바꿔야 한다
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_ap = preprocess_input(arr_ap)
print(arr_dog.shape) # (224, 224, 3) # shape는 같지만 3 부분이, RGB->BGR로 바뀜


arr_input = np.stack([arr_dog, arr_cat, arr_ap])
print(arr_input.shape) # (2, 224, 224, 3)


# 모델
model = VGG16(weights='imagenet')
probs = model.predict(arr_input)

print(probs)
print(probs.shape) # (2, 1000)

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions
result = decode_predictions(probs)
print('----------------------------')
print('result[0]',result[0])
print('----------------------------')
print('result[1]',result[1])
print('----------------------------')
print('result[2]',result[2])
print('----------------------------')
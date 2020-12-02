import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

# 1.데이터
# 1.1 load_data
# 1.2 train_test_split
# 1.3 scaler
# 1.4 reshape
# 2.모델
# 3.컴파일 훈련
# 4.평가 예측


import numpy as np

# 1.1 load_data
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("origin x shape:",x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print("origin y shape:",y_train.shape, y_test.shape) # (60000,) (10000,)


# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)
print(y_train[0]) # [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]


# 1.2 train_test_split
# load_data할 때 이미 train과 test가 나뉘어 있으니 별도로 나누지 않는다
# validation은 fit에서 validation_split으로 적용한다


# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train) # fit하고
# x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자

# Scaler는 2차원 이하만 된다, 수동으로 바꾸자 (게다가 최소/최대값을 알고 있으니...)
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


# 1.4 reshape
# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
print("reshape x:", x_train.shape, x_test.shape)



# 2.모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

def build_model(drop=0.5, optimizer='adam', node_value=64, layer_num=1):
    inputs = Input( shape=(28*28, ) )
    for cnt in range(layer_num):
        x = Dense(node_value, activation='relu', name='hidden1')(inputs)
        x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, 
                    metrics=['acc'],
                    loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    epochs = [10, 20, 30]
    node_value = [64, 128, 256]
    layer_num = [1,2,3]
    return_parameter = {"batch_size":batches, 
                        "optimizer":optimizers, 
                        "drop":dropout,
                        "epochs":epochs,
                        "node_value":node_value,
                        "layer_num":layer_num
                        }
    return return_parameter

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

wrapper_model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()
# search = GridSearchCV(build_model, hyperparameters, cv=3) # fit에 문제가 생긴다
# search = GridSearchCV(wrapper_model, hyperparameters, cv=3) # wrapper를 씌워 사이킷런으로 가져온다
search = RandomizedSearchCV(wrapper_model, hyperparameters, cv=3) # wrapper를 씌워 사이킷런으로 가져온다

search.fit(x_train, y_train)

print(search.best_params_)
acc = search.score(x_test, y_test)
print("최종 스코어:", acc)

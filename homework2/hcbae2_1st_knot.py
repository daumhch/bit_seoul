# keras66부터 73까지 모두 적용
# 하이퍼 파라미터 튜닝 및
# 레이어 구성 옵션들

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("origin x shape:",x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print("origin y shape:",y_train.shape, y_test.shape) # (60000,) (10000,)



# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)



# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print("reshape x_train:", x_train.shape)
print("reshape x_test:", x_test.shape)



# Scaler
# 선택은 아무거나, 최적이라 생각하는 주관적 판단
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
# print(x_train[0])



# 2.모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ReLU, ELU, LeakyReLU, PReLU
from tensorflow.keras.activations import relu, selu, elu, softmax, sigmoid

from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

def build_model(layer_num1=1,
                node_value1=64,
                activation1=relu,
                drop1=0.2,

                layer_num2=1,
                node_value2=64,
                activation2=ReLU,
                drop2=0.2,

                optimizer=Adam,
                learning_rate=0.001
                ):
    inputs = Input( shape=(28, 28, 1), )
    ### Conv2D
    for cnt in range(layer_num1):
        if cnt == 0:
            x = Conv2D(node_value1, (3,3), padding='same')(inputs)
        else:
            x = Conv2D(node_value1, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation1)(x)
        x = Dropout(drop1)(x)
    x = Flatten()(x)

    ### Flatten 이후 Dense
    for cnt2 in range(layer_num2):
        x = Dense(node_value2)(x)
        x = activation2()(x)
        x = Dropout(drop2)(x)
    
    # 함수형 모델 마무리
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(lr=learning_rate), 
                    metrics=['acc'],
                    loss='categorical_crossentropy')

    # 모델 정보 출력
    print("optimizer: ",optimizer, "/ learning_rate:",learning_rate)
    print("layer_num1:",layer_num1, " / node_value1:",node_value1)
    print("activation1: ",activation1)
    print("layer_num2:",layer_num2, " / node_value1:",node_value2)
    print("activation2: ",activation2)
    print(model.summary() )
    return model

def create_hyperparameters():
    epochs = [20, 40, 60]

    layer_num1=[1,2,3]
    node_value1=[64,128,256]
    activation1=[relu, selu, elu]
    drop1=[0.2, 0.35, 0.5]

    layer_num2=[1,2,3]
    node_value2=[64,128,256]
    activation2=[ReLU, ELU, LeakyReLU]
    drop2=[0.2, 0.35, 0.5]

    optimizer = [Adam, RMSprop, SGD]
    learning_rate= [0.001, 0.01, 0.1]

    return_parameter = {"epochs":epochs,

                    "layer_num1":layer_num1, 
                    "node_value1":node_value1,
                    "activation1":activation1,
                    "drop1":drop1,

                    "layer_num2":layer_num2, 
                    "node_value2":node_value2,
                    "activation2":activation2,
                    "drop2":drop2,

                    "optimizer":optimizer,
                    "learning_rate":learning_rate
                    }
    return return_parameter




from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

wrapper_model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()
search = RandomizedSearchCV(wrapper_model, hyperparameters, cv=3)

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(monitor='loss',
                                patience=5,
                                mode='auto')
search.fit(x_train, y_train, 
            callbacks=[early_stopping])

print("search.best_params_:\n",search.best_params_)
acc = search.score(x_test, y_test)
print("최종 스코어:", acc)


# search.best_params_:
#  {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.gradient_descent.SGD'>, 'node_value2': 256, 'node_value1': 256, 'learning_rate': 0.001, 'layer_num2': 3, 'layer_num1': 2, 'epochs': 20, 'drop2': 0.2, 'drop1': 0.35, 'activation2': <class 'tensorflow.python.keras.layers.advanced_activations.ELU'>, 'activation1': <function selu at 0x00000215469B3790>}
# 313/313 [==============================] - 2s 5ms/step - loss: 0.0497 - acc: 0.9859
# 최종 스코어: 0.9858999848365784
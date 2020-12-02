# activation 추가하기

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기




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
from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import relu, selu, elu, softmax, sigmoid
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import ReLU, ELU

def build_model(drop=0.5, 
                optimizer=Adam, 
                learning_rate_num=0.001,
                node_value=64, 
                layer_num=1,
                activations=ReLU):
                # activations=relu 일 때에는 Activation(relu) 또는 Activation('relu')
                # activations=ReLU 일 때에는 activation(여러 옵션들)

    inputs = Input( shape=(28*28, ) )
    for cnt in range(layer_num):
        if cnt == 0:
            x = Dense(node_value, activation='relu')(inputs)
        else :
            x = Dense(node_value, activation='relu')(x)
        x = activations()(x)
        x = Dense(node_value)(x)
        x = Activation(relu)(x)
        x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(lr=learning_rate_num), 
                    metrics=['acc'],
                    loss='categorical_crossentropy')
    print("optimizer: ",optimizer,"/ learning_rate_num:",learning_rate_num)
    print("node_value:",node_value,"/ layer_num:",layer_num)
    print("activation: ",activations)
    print(model.summary() )
    return model


def create_hyperparameters():
    batches = [10]
    optimizers = [Adam, RMSprop]
    learning_rate_num = [0.001, 0.01]
    dropout = [0.2, 0.5]
    epochs = [20]
    node_value = [10, 20]
    layer_num = [1, 2, 3]
    activations = [ReLU, ELU, LeakyReLU]
    
    return_parameter = {"batch_size":batches, 
                        "optimizer":optimizers, 
                        "learning_rate_num":learning_rate_num,
                        "drop":dropout,
                        "epochs":epochs,
                        "node_value":node_value,
                        "layer_num":layer_num,
                        "activations":activations
                        }
    return return_parameter

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

wrapper_model = KerasClassifier(build_fn=build_model, verbose=0)
hyperparameters = create_hyperparameters()
# search = GridSearchCV(build_model, hyperparameters, cv=3) # fit에 문제가 생긴다
# search = GridSearchCV(wrapper_model, hyperparameters, cv=3) # wrapper를 씌워 사이킷런으로 가져온다
search = RandomizedSearchCV(wrapper_model, hyperparameters, cv=3) # wrapper를 씌워 사이킷런으로 가져온다


early_stopping = EarlyStopping(monitor='loss',
                                patience=5,
                                mode='auto')
search.fit(x_train, y_train, 
            callbacks=[early_stopping])

print("search.best_params_:\n",search.best_params_)
acc = search.score(x_test, y_test)
print("최종 스코어:", acc)







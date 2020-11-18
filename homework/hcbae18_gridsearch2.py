import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


# 보스턴의 집값을 예측하는 데이터셋 예제

'''
x
506 행 13 열 
CRIM     per capita crime rate by town
ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS    proportion of non-retail business acres per town
CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX      nitric oxides concentration (parts per 10 million)
RM       average number of rooms per dwelling
AGE      proportion of owner-occupied units built prior to 1940
DIS      weighted distances to five Boston employment centres
RAD      index of accessibility to radial highways
TAX      full-value property-tax rate per $10,000
PTRATIO  pupil-teacher ratio by town
B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT    % lower status of the population

y
506 행 1 열
target (MEDV)     Median value of owner-occupied homes in $1000's
'''

import numpy as np
from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target

print("x.shape:",x.shape) # (506, 13)
print("y.shape:",y.shape) # (506, )


# 구현 순서
# 데이터 전처리
# 모델
# 컴파일 훈련
# 평가 예측
# 회귀는 RMSE와 R2

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x) # fit하고
x = scaler.transform(x) # 사용할 수 있게 바꿔서 저장하자





# train/test/val
from sklearn.model_selection import train_test_split 
x_train,x_rest, y_train,y_rest = train_test_split(
    x, y, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x_test,x_val, y_test,y_val = train_test_split(
    x_rest,y_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔

print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)
print("after x_val.shape",x_val.shape)



# reshape?
# x가 2차원이라, CNN을 위해 3차원으로 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test = x_test.reshape(x_test.shape[0],x_train.shape[1],1,1)
x_val = x_val.reshape(x_val.shape[0],x_train.shape[1],1,1)
print("reshape x:", x_train.shape, x_test.shape, x_val.shape)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

def create_model():
    # 2. 모델 구성
    model = Sequential()
    model.add(Conv2D(32, (x_train.shape[1]), padding='same', input_shape=(x_train.shape[1],1,1) ))
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(1))
    model.summary()

    # 3. 컴파일, 훈련
    model.compile(
        loss='mse',
        optimizer='adam')
    return model


from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import log_loss, make_scorer

model = KerasRegressor(build_fn=create_model, verbose=1)

batch_size = [128, 256, 512, 1024]
param_grid = dict(batch_size=batch_size, epochs=[1000])
grid = GridSearchCV(
    estimator=model, 
    param_grid=param_grid,
    cv= 2,
    n_jobs=1)
grid_result = grid.fit(x_train, y_train,validation_data=(x_val,y_val) )
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))




print("hcbae18_gridsearch2 end")

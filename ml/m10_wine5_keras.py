import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



import pandas as pd
import numpy as np

wine = pd.read_csv('./data/csv/winequality-white.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=';' # 구분 기호
                        )


y = wine['quality']
x = wine.drop('quality', axis=1).to_numpy()
print(x.shape)
print(y.shape)

newlist = []
for i in list(y):
    if i <=4:
        newlist += [0]
    elif i <=7:
        newlist += [1]
    else:
        newlist += [2]

y = newlist

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# 카테고리컬 하면 0부터 시작함, 그래서 결과가 0~9까지 10개 출력됨

print(x.shape)

# y의 범위를 3~9 -> 0,1,2로 좁히는 것은
# 평가 방법을 바꾼 것이지 데이터 조작이 아니다




# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


# 1.3 scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print("x_train.shape",x_train.shape)
print("x_test.shape",x_test.shape)

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score


# 1.4 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)



# 2.모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import GRU, Conv1D
from tensorflow.keras.layers import Dropout, Flatten, MaxPooling1D
model = Sequential()
model.add(Conv1D(64, 3, 
                padding='same',
                strides=1,
                activation='relu',
                input_shape=(x_train.shape[1],1) ))
# model.add(Conv1D(64, 9, padding='same', strides=1, activation='relu') )
model.add(MaxPooling1D(pool_size=2, padding='valid', strides=2))
model.add(Dropout(0.2))

model.add(Conv1D(128, 3, padding='same', strides=1, activation='relu') )
# model.add(Conv1D(128, 9, padding='same', strides=1, activation='relu') )
model.add(MaxPooling1D(pool_size=2, padding='valid', strides=2))
model.add(Dropout(0.3))

model.add(Conv1D(256, 3, padding='same', strides=1, activation='relu') )
# model.add(Conv1D(256, 9, padding='same', strides=1, activation='relu') )
model.add(MaxPooling1D(pool_size=2, padding='valid', strides=2))
model.add(Dropout(0.4))
model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
# model.add(Dense(512, activation = 'relu'))
# model.add(Dense(512, activation = 'relu'))
# model.add(Dense(512, activation = 'relu'))
model.add(Dense(3, activation = 'softmax') )
model.summary()



# 3.훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    mode='auto',
    verbose=2)


hist = model.fit(
    x_train, y_train,
    epochs=10000,
    batch_size=128,
    verbose=1,
    validation_split=0.5,
    callbacks=[early_stopping])



# 4.평가 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("accuracy: ", accuracy)

y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
# print("y_test:\n", y_test)
# print("y_predict:\n", y_predict)


import matplotlib.pyplot as plt
plt.subplot(2,1,1) # 2장 중에 첫 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2) # 2장 중에 두 번째
plt.plot(hist.history['accuracy'], marker='.', c='red')
plt.plot(hist.history['val_accuracy'], marker='.', c='blue')
plt.grid()
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['accuracy', 'val_accuracy'])

plt.show()

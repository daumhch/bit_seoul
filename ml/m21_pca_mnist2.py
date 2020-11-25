# PCA로 축소해서 DNN 모델을 완성하시오
# 1. 0.95이상
# 2. 1이상
# mnist DNN과 loss와 acc를 비교하시오


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


# 1.데이터
# 1.1 load_data
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
print("x.shape:", x.shape) # (70000, 28, 28)

x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
print("after reshape x.shape:", x.shape) # (70000, 784)

# 1.5 PCA
from sklearn.decomposition import PCA
cumsum_standard = 0.95
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= cumsum_standard) +1
print("n_components:",d) # 154

pca = PCA(n_components=d) # 소숫점을 넣어도 적용된다
x = pca.fit_transform(x)
print("after pca x.shape", x.shape) # (70000, 154)




# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)



# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, train_size=60000, test_size=10000)
print("split shape:",x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)



# 1.4 reshape
# PCA를 위해서 위에서 했음




# 이미지를 DNN 할 때에는 reshape하고 scaler 하자
# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) # fit하고
x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자
x_test = scaler.transform(x_test) # 사용할 수 있게 바꿔서 저장하자
# x_train = x_train.astype('float32')/255.
# x_test = x_test.astype('float32')/255.






# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],) ))
model.add(Dense(64, activation='relu') )
model.add(Dense(10, activation='softmax') )
model.summary()



# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='loss',
    patience=20,
    mode='auto',
    verbose=2)


model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=128,
    verbose=0,
    validation_split=0.2,
    callbacks=[early_stopping])



# 4. 평가, 예측

loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("accuracy: ", accuracy)








import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기




import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # mnist 데이터 셋 로드

print("origin x shape:",x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print("origin y shape:",y_train.shape, y_test.shape) # (60000,) (10000,)



# OneHotEncoding
# 케라스에서 지원하는 one hot Encoding
# [1,5,2,8,2,5,...] -> 각 숫자를 카테고리화 하여 2진 배열값으로 바꾼다
from tensorflow.keras.utils import to_categorical 
print("before y_train[0]:",y_train[0]) # 5
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (60000,10) (10000,10)
print("after y_train[0]:",y_train[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]




# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
# shape 뒷부분이 28, 28 일 필요는 없지만, 일단 그대로 가져다 사용한다
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print("reshape x:", x_train.shape, x_test.shape)



# Scaler
# 선택은 아무거나, 최적이라 생각하는 주관적 판단
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
# print(x_train[0])




# 배운 방법으로 스케일링 해보자
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train) # fit하고
# x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자
# x_test = scaler.transform(x_test) # x_train fit 한걸 x_test에도 적용
# 하려고 했으나, Scaler는 2차원 데이터를 받고 싶어한다
# 스케일러 쓰려고 3차원->2차원->스케일러->3차원 하느니
# 그냥 선생님 방식대로 나누는 것이 구조가 간단할 것 같다




# 10000개 데이터 중에 99980개는 validation, 20개는 test
# train_size=0.002, test_size=0.998
from sklearn.model_selection import train_test_split 
x_test,x_val, y_test,y_val = train_test_split(
    x_test,y_test, train_size=0.1, test_size=0.9)

print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)
print("after x_val.shape",x_val.shape)

def make_model(node_1st, y_test):
    # 2.모델
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

    model = Sequential()
    model.add( Conv2D(node_1st[0], (2,2), padding='same', input_shape=(28,28,1) ) )
    model.add(MaxPooling2D(pool_size=(2,2) ))
    model.add( Conv2D(node_1st[0], (2,2) ) )
    model.add(MaxPooling2D(pool_size=(2,2) ))
    model.add(Flatten())
    model.add(Dense(node_1st[1]))
    model.add(Dense(10, activation='softmax') )
    model.summary()


    # 3. 컴파일, 훈련
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )

    from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=3,
        mode='auto',
        verbose=2)

    model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=0,
        validation_data=(x_val,y_val),
        callbacks=[early_stopping])



    # 4. 평가, 예측
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
    # print("loss: ", loss)
    print("accuracy: ", accuracy)


    y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
    y_test = np.argmax(y_test, axis=1)
    y_predict = np.argmax(y_predict, axis=1)
    # print("y_predict:\n", y)
    # print("y_test:\n", y_test)
    # print("y_predict:\n", y_predict)
    return accuracy
# end of make_model



result_list = np.zeros((10,10))

for cnt1 in range(1,10):
    for cnt2 in range(1,10):
        result = make_model([cnt1,cnt2], y_test)
        result_list[cnt1][cnt2] = result


print("???:",result_list )
print("???:", np.max(result_list) )



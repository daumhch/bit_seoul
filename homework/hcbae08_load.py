import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
dataset = np.array(range(1,101))
size = 5


def hcbae_split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        # [item~~~] 이 없어도 되는데, 있는 이유가 있겠지?
        # aaa.insert(i, subset)
        aaa.append(subset)
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자


datasets = hcbae_split_x(dataset,size)

x = datasets[:, :4]
y = datasets[:, 4]
print("x.shape:", x.shape) # (96,4)

x = np.reshape(x, (x.shape[0], x.shape[1], 1))
print("x.reshape:", x.shape) # (96,4,1)


from sklearn.model_selection import train_test_split 
x_train,x_rest, y_train,y_rest = train_test_split(
    x, y, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x_test,x_val, y_test,y_val = train_test_split(
    x_rest,y_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔

print("x_train.reshape:", x_train.shape) # (57,4,1)


# 2.모델
# model = Sequential()
# model.add(LSTM(100, input_shape=(4,1)))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

model = load_model("./save/hcbae_model_save.h5")
model.add(Dense(5))
model.add(Dense(1))

model.summary()




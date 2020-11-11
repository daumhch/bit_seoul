# 1.데이터
import numpy as np
x1 = np.array([range(1,101),range(711,811), range(100)])
y1 = np.array([range(101,201), range(311,411), range(100)])
x1 = x1.T
y1 = y1.T
print(x1.shape)

x2 = np.array([range(4,104),range(761,861), range(100)])
y2 = np.array([range(501,601), range(431,331), range(100,200)])
x2 = x2.T
y2 = y2.T
print(x2.shape)


# 2. 모델구성
from tensorflow.keras.layers import Dense # Dense layer를 사용
from tensorflow.keras.models import Model # 함수형 모델 사용
from tensorflow.keras.layers import Input # 함수형 모델에는 input layer가 따로 있다

input1 = Input(shape=(3,))
dense1_1 = Dense(5, activation='relu')(input1)
dense1_2 = Dense(4, activation='relu')(dense1_1)
dense1_3 = Dense(3, activation='relu')(dense1_2)
output1 = Dense(1)(dense1_3)
model1 = Model(inputs=input1, outputs=output1)
model1.summary()

input2 = Input(shape=(3,))
dense2_1 = Dense(5, activation='relu')(input2)
dense2_2 = Dense(4, activation='relu')(dense2_1)
dense2_3 = Dense(3, activation='relu')(dense2_2)
output2 = Dense(1)(dense2_3)
model2 = Model(inputs=input2, outputs=output2)
model2.summary()








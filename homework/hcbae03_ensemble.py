import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

import numpy as np
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import Model # 함수형 모델 사용
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Concatenate, concatenate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



#재현을 위한 랜덤시드 고정
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from tensorflow.python.keras import backend as K
tf.random.set_seed(1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)




# 인풋 2개, 아웃풋 1개 모델링 해보자

# 1.데이터
x1 = np.array([range(1,101),range(711,811), range(100)])
x2 = np.array([range(4,104),range(761,861), range(100)])

y1 = np.array([range(101,201), range(311,411), range(100)])

x1 = x1.T
x2 = x2.T
print(x1.shape)
print(x2.shape)

y1 = y1.T


x1_train, x1_rest, y1_train, y1_rest = train_test_split(
    x1, y1, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x1_test, x1_val, y1_test, y1_val = train_test_split(
    x1_rest, y1_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔

# 3개의 데이터 까지 나눌 수 있다
x2_train, x2_rest, = train_test_split(
    x2, train_size=0.6, test_size=0.4) # 6:4로 먼저 나누고
x2_test, x2_val, = train_test_split(
    x2_rest, train_size=0.5, test_size=0.5) # 남은 4를 5:5로 나눔


print("y1_train.size", y1_train.size)
print("y1_val.size", y1_val.size)
print("y1_test.size", y1_test.size)




# 2. 모델구성

#내 가정, 최적화를 하면, epochs 숫자를 줄여도 될 것이다

'''
                          mean / median
epochs 50 = 39691.50175537109 / 659.1607360839844
            39691.502318115236 / 659.1606750488281
epochs 100 = 19878.369839022158 / 278.0535430908203
             19878.369788998367 / 278.0534973144531
epochs 250 = 7952.48243412292 / 1.80262291431427
             7952.486722365767 / 1.8013672232627869
epochs 500 = 3978.7956265942826 / 0.8127634227275848
             3979.1836526859774 / 1.361957609653473
epochs 750 = 45715.73813848818 / 0.5890607535839081
'''

# epochs 250 고정일 때
# output_dense = [1000, 1000, 1000, 1000, 1000] 일 때
# input_dense = [1000, 1000, 1000, 1000, 1000] # 7952.486722365767
# input_dense = [500, 750, 1000, 1250, 1500] # 26429.39870183149
# input_dense = [1500, 1250, 1000, 750, 500] # 11159.743827445209

# epochs 250 고정일 때
# input_dense = [1000, 1000, 1000, 1000, 1000] 일 때
# output_dense = [1000, 1000, 1000, 1000, 1000] # 7952.486722365767
# output_dense = [500, 750, 1000, 1250, 1500] # 14166.563491895467
# output_dense = [1500, 1250, 1000, 750, 500] # 9681.130631741285

# epochs 250 고정일 때
# 다이아몬드 모양
# input_dense = [500, 750, 1000, 1250, 1500] 
# output_dense = [1500, 1250, 1000, 750, 500]
# 11747.927622785031 / 6.785780429840088 의외로 안 좋다? 


# epochs 250 고정일 때
# 전체적인 역삼각형
input_dense = [1500, 1250, 1000, 750, 500] 
output_dense = [250, 125, 62, 31, 15]
# 3436.7736694196315 / 6.96921968460083 이게 왠지 제일 낫다?
# 3431.303926724434 / 5.888979434967041 #집에서 돌린 값 1

# input_dense = [1500, 1250, 1000, 750, 500] 
# output_dense = [500, 125, 62, 31, 15]
# 5010.853225636184 / 4.4073827266693115 위에보다 평균이 나빠졌다?

epochs_num = 250

batch_size_num = 32

# 모델 1
input1 = Input(shape=(3,))
dense1_1 = Dense(input_dense[0], activation='relu', name='dense1_1')(input1)
dense1_2 = Dense(input_dense[1], activation='relu', name='dense1_2')(dense1_1)
dense1_3 = Dense(input_dense[2], activation='relu', name='dense1_3')(dense1_2)
dense1_4 = Dense(input_dense[3], activation='relu', name='dense1_4')(dense1_3)
output1 = Dense(input_dense[4], name='output1')(dense1_4)
# model1 = Model(inputs=input1, outputs=output1)
# model1.summary()

# 모델 2
input2 = Input(shape=(3,))
dense2_1 = Dense(input_dense[0], activation='relu', name='dense2_1')(input2)
dense2_2 = Dense(input_dense[1], activation='relu', name='dense2_2')(dense2_1)
dense2_3 = Dense(input_dense[2], activation='relu', name='dense2_3')(dense2_2)
dense2_4 = Dense(input_dense[3], activation='relu', name='dense2_4')(dense2_3)
output2 = Dense(input_dense[4], name='output2')(dense2_4)
# model2 = Model(inputs=input2, outputs=output2)
# model2.summary()

# 모델 병합
# 대문자 소문자 본질적으로 같은데, 사용 방법이 다르다

# merge1 = concatenate([output1, output2]) # concatenate layer
# merge1 = Concatenate()([output1, output2])
merge1 = Concatenate(axis=1)([output1, output2])
# middle1 = Dense(256, name='middle1')(merge1)
# middle2 = Dense(256, name='middle2')(middle1)
# middle3 = Dense(256, name='middle3')(middle2)

# 이렇게 해도 된다
# middle3 = Dense(30)(merge1)
# middle3 = Dense(7)(middle3)
# middle3 = Dense(11)(middle3)

# output 모델 구성(분기)
# middle layer 하기 싫으면 merge1을 바로 입력하면 된다
output1_1 = Dense(output_dense[0], name='output1_1')(merge1)
output1_2 = Dense(output_dense[1], name='output1_2')(output1_1)
output1_3 = Dense(output_dense[2], name='output1_3')(output1_2)
output1_4 = Dense(output_dense[3], name='output1_4')(output1_3)
output1_5 = Dense(output_dense[4], name='output1_5')(output1_4)
output1_6 = Dense(3)(output1_5)


# 모델 정의
model = Model(inputs=[input1,input2], 
                outputs=output1_6)
# model.summary()



# 3. 컴파일, 훈련
model.compile( # 컴파일
    loss='mse', # 오차함수는 mean squared error를 사용한다
    optimizer='adam', # 최적화 방법은 'adam'을 사용한다
    metrics=['mae']) # mean absolute error 지표를 추가한다

#훈련, 일단 x_train, y_train 입력하고
hist = model.fit([x1_train, x2_train], y1_train, 
    epochs=epochs_num, # 훈련 횟수
    batch_size=batch_size_num, # 훈련 데이터단위
    validation_data=([x1_val, x2_val], y1_val), # 검증 데이터 사용하기
    verbose=0)

# 4. 평가, 예측
# 평가 데이터 넣어서 결과 보기
result = model.evaluate([x1_test,x2_test], y1_test, batch_size=batch_size_num) 
print("result : ", result)

# output이 3개 -> loss 3개, mse3개가 출력되고, 전체 loss 포함해서 7개 출력
# output 별로 loss와 mse를 보고, 성능 개선 이유를 찾을 수 있다
# 전체 loss = output들의 loss 합


y1_predict = model.predict([x1_test,x2_test]) # 평가 데이터 다시 넣어 예측값 만들기
# print("y_predict:\n", y_predict)



# 사용자정의 RMSE 함수
# 사이킷런의 metrics에서 mean_squared_error을 불러온다
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y1_test, y1_predict))



# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
r2 = r2_score(y1_test, y1_predict)
print("R2:", r2)


# 고정 값 커스텀 대입
x1_demo = np.full((1,3),[1,711,0])
x2_demo = np.full((1,3),[4,761,0])
y1_val_predict = model.predict([x1_demo,x2_demo]) # 평가 데이터 다시 넣어 예측값 만들기
print("fix x1_demo, x2_demo:\n", x1_demo, x2_demo)
print("fix y1[0]:\n", y1[0])
print("predict y1_predict: %\n", y1_val_predict)





print("find loss mean/median: %\n", np.mean(hist.history['loss']), "/", np.median(hist.history['loss']) )


# 그래프 그리기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()


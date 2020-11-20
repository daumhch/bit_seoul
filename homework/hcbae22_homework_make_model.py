import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


view_size = 3

def split_x2(seq, size):
    bbb = []
    for i in range(len(seq) - size + 1):
        bbb.append(seq[i:(i+size)])
    return np.array(bbb)

# 1.데이터
# 1.1 load_data
# 1.2 train_test_split
# 1.3 scaler
# 1.4 reshape
# 2.모델
# 3.컴파일 훈련
# 4.평가 예측


# 1. 데이터
# 1.1 load_data
import numpy as np
samsung_data = np.load('./data/samsung_data.npy', allow_pickle=True)
samsung_data = split_x2(samsung_data, view_size)
samsung_data = samsung_data[:-1]
# print("type(samsung_data):",type(samsung_data))
# print("samsung_data:",samsung_data)
# print("samsung_data.shape:", samsung_data.shape)
samsung_target = np.load('./data/samsung_target.npy', allow_pickle=True)
samsung_target = samsung_target[view_size:]
# print("samsung_target:",samsung_target)
# print('samsung_target.shape',samsung_target.shape)


bitcom_data = np.load('./data/bitcom_data.npy', allow_pickle=True)
bitcom_data = split_x2(bitcom_data, view_size)
bitcom_data = bitcom_data[:-1]
# print("type(bitcom_data):",type(bitcom_data))
# print("bitcom_data:",bitcom_data)
# print("bitcom_data.shape:", bitcom_data.shape)
bitcom_target = np.load('./data/bitcom_target.npy', allow_pickle=True)
bitcom_target = bitcom_target[view_size:]
# print("bitcom_target:",bitcom_target)
# print('bitcom_target.shape',bitcom_target.shape)


print("========== 데이터 로딩 끝 ==========")
print("samsung_data.shape:", samsung_data.shape)
print('samsung_target.shape',samsung_target.shape)
print("bitcom_data.shape:", bitcom_data.shape)
print('bitcom_target.shape',bitcom_target.shape)


# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
samsung_data_train,samsung_data_test, samsung_target_train,samsung_target_test = train_test_split(
    samsung_data, samsung_target, train_size=0.9, test_size=0.1, random_state = 44)
# print("after samsung_data_train.shape:\n",samsung_data_train.shape)
# print("after samsung_data_test.shape:\n",samsung_data_test.shape)
# print("samsung_data_train[0]:\n",samsung_data_train[0])
# print("samsung_data_test[0]:\n",samsung_data_test[0])
bitcom_data_train,bitcom_data_test, bitcom_target_train,bitcom_target_test = train_test_split(
    bitcom_data, bitcom_target, train_size=0.9, test_size=0.1, random_state = 44)
# print("after bitcom_data_train.shape:\n",bitcom_data_train.shape)
# print("after bitcom_data_test.shape:\n",bitcom_data_test.shape)
# print("bitcom_data_train[0]:\n",bitcom_data_train[0])
# print("bitcom_data_test[0]:\n",bitcom_data_test[0])



def scaling3D(data, scaler):
    temp_data = data
    num_sample   = data.shape[0] # 면
    num_sequence = data.shape[1] # 행
    num_feature  = data.shape[2] # 열
    for ss in range(num_sequence):
        scaler.fit(data[:, ss, :])

    results = []
    for ss in range(num_sequence):
        results.append(scaler.transform(data[:, ss, :]).reshape(num_sample, 1, num_feature))
    temp_data = np.concatenate(results, axis=1)
    return temp_data

def transform3D(data, scaler):
    temp_data = data
    num_sample   = data.shape[0] # 면
    num_sequence = data.shape[1] # 행
    num_feature  = data.shape[2] # 열
    results = []
    for ss in range(num_sequence):
        results.append(scaler.transform(data[:, ss, :]).reshape(num_sample, 1, num_feature))
    temp_data = np.concatenate(results, axis=1)
    return temp_data


# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
samsung_scaler = StandardScaler()
samsung_data_train = scaling3D(samsung_data_train, samsung_scaler)
samsung_data_test = transform3D(samsung_data_test, samsung_scaler)
print("after scaled samsung_data_train.shape:",samsung_data_train.shape)
print("after scaled samsung_data_test.shape:",samsung_data_test.shape)
# print("after scaled samsung_data_train[0]:",samsung_data_train[0])
# print("after scaled samsung_data_test[0]:",samsung_data_test[0])

bitcom_scaler = StandardScaler()
bitcom_data_train = scaling3D(bitcom_data_train, bitcom_scaler)
bitcom_data_test = transform3D(bitcom_data_test, bitcom_scaler)
print("after scaled bitcom_data_train.shape",bitcom_data_train.shape)
print("after scaled bitcom_data_test.shape:",bitcom_data_test.shape)
# print("after scaled bitcom_data_train[0]:",bitcom_data_train[0])
# print("after scaled bitcom_data_test[0]:",bitcom_data_test[0])


# 1.4 reshape




# modelpath = './model/hcbae22_{epoch:02d}_{val_loss:.4f}.hdf5'
# model_save_path = "./save/hcbae22_model.h5"
# weights_save_path = './save/hcbae22_weights.h5'

# 2.모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

samsung_model = Sequential()
samsung_model.add(Dense(30, name='samsung_model_1', activation='relu', input_shape=(samsung_data_train.shape[1],) ))
samsung_model.add(Dense(30, name='samsung_model_2', activation='relu') )
samsung_model.add(Dense(20, name='samsung_model_3', activation='relu') )
samsung_model.add(Dense(1, name='samsung_model_4', activation='relu') )
samsung_model.summary()

bitcom_model = Sequential()
bitcom_model.add(Dense(30, name='bitcom_model_1', activation='relu', input_shape=(bitcom_data_train.shape[1],) ))
bitcom_model.add(Dense(30, name='bitcom_model_2', activation='relu') )
bitcom_model.add(Dense(20, name='bitcom_model_3', activation='relu') )
bitcom_model.add(Dense(1, name='bitcom_model_4', activation='relu') )
bitcom_model.summary()

from tensorflow.keras.layers import Concatenate, concatenate
merge1 = Concatenate(axis=1)([samsung_model.output, bitcom_model.output])
middle = Dense(30, name='middle1')(merge1)
middle = Dense(7, name='middle2')(middle)
middle = Dense(11, name='middle3')(middle)

samsung_out = Dense(30, name='output1_1')(middle)
samsung_out = Dense(7, name='output1_2')(samsung_out)
samsung_out = Dense(1, name='output1_3')(samsung_out)

bitcom_out = Dense(15, name='output2_0')(middle)
bitcom_out = Dense(17, name='output2_1')(bitcom_out)
bitcom_out = Dense(11, name='output2_3')(bitcom_out)
bitcom_out = Dense(1, name='output2_4')(bitcom_out)


total_model = Model(inputs=[samsung_model.input,bitcom_model.input], 
                    outputs=[samsung_out,bitcom_out])
total_model.summary()



# samsung_data_train = scaling3D(samsung_data_train, samsung_scaler)
# samsung_data_test = transform3D(samsung_data_test, samsung_scaler)
# bitcom_data_train = scaling3D(bitcom_data_train, bitcom_scaler)
# bitcom_data_test
# 3. 컴파일, 훈련
total_model.compile( # 컴파일
    loss='mse', # 오차함수는 mean squared error를 사용한다
    optimizer='adam', # 최적화 방법은 'adam'을 사용한다
    metrics=['mae']) # mean absolute error 지표를 추가한다

#훈련, 일단 x_train, y_train 입력하고
total_model.fit([samsung_data_train, bitcom_data_train], 
    epochs=256, # 훈련 횟수
    batch_size=32, # 훈련 데이터단위
    validation_split=0.2,
    verbose=1)
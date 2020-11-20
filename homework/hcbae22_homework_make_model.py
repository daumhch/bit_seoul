import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


view_size = 10

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
samsung_data = np.load('./data/samsung_data.npy', allow_pickle=True).astype('float32')
samsung_data = split_x2(samsung_data, view_size)
samsung_pred = samsung_data[-1]
# print("samsung_pred:\n",samsung_pred)

samsung_data = samsung_data[:-1]
# print("type(samsung_data):",type(samsung_data))
# print("samsung_data:",samsung_data)
# print("samsung_data.shape:", samsung_data.shape)
samsung_target = np.load('./data/samsung_target.npy', allow_pickle=True).astype('float32')
samsung_target = samsung_target[view_size:]
# print("samsung_target:",samsung_target)
# print('samsung_target.shape',samsung_target.shape)


bitcom_data = np.load('./data/bitcom_data.npy', allow_pickle=True).astype('float32')
bitcom_data = split_x2(bitcom_data, view_size)
bitcom_pred = bitcom_data[-1]
# print("bitcom_pred:\n",bitcom_pred)

bitcom_data = bitcom_data[:-1]
# print("type(bitcom_data):",type(bitcom_data))
# print("bitcom_data:",bitcom_data)
# print("bitcom_data.shape:", bitcom_data.shape)
bitcom_target = np.load('./data/bitcom_target.npy', allow_pickle=True).astype('float32')
bitcom_target = bitcom_target[view_size:]
# print("bitcom_target:",bitcom_target)
# print('bitcom_target.shape',bitcom_target.shape)


gold_data = np.load('./data/gold_data.npy', allow_pickle=True).astype('float32')
gold_data = split_x2(gold_data, view_size)
gold_pred = gold_data[-1]
# print("gold_pred:\n",gold_pred)

gold_data = gold_data[:-1]
# print("type(gold_data):",type(gold_data))
# print("gold_data:",gold_data)
# print("gold_data.shape:", gold_data.shape)
gold_target = np.load('./data/gold_target.npy', allow_pickle=True).astype('float32')
gold_target = gold_target[view_size:]
# print("gold_target:",gold_target)
# print('gold_target.shape',gold_target.shape)



kosdaq_data = np.load('./data/kosdaq_data.npy', allow_pickle=True).astype('float32')
kosdaq_data = split_x2(kosdaq_data, view_size)
kosdaq_pred = kosdaq_data[-1]
# print("kosdaq_pred:\n",kosdaq_pred)

kosdaq_data = kosdaq_data[:-1]
# print("type(kosdaq_data):",type(kosdaq_data))
# print("kosdaq_data:",kosdaq_data)
# print("kosdaq_data.shape:", kosdaq_data.shape)
kosdaq_target = np.load('./data/kosdaq_target.npy', allow_pickle=True).astype('float32')
kosdaq_target = kosdaq_target[view_size:]
# print("kosdaq_target:",kosdaq_target)
# print('kosdaq_target.shape',kosdaq_target.shape)



print("========== 데이터 로딩 끝 ==========")
print("samsung_data.shape:", samsung_data.shape)
print('samsung_target.shape',samsung_target.shape)
print("bitcom_data.shape:", bitcom_data.shape)
print('bitcom_target.shape',bitcom_target.shape)
print("gold_data.shape:", gold_data.shape)
print('gold_target.shape',gold_target.shape)
print("kosdaq_data.shape:", kosdaq_data.shape)
print('kosdaq_target.shape',kosdaq_target.shape)

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

gold_data_train,gold_data_test, gold_target_train,gold_target_test = train_test_split(
    gold_data, gold_target, train_size=0.9, test_size=0.1, random_state = 44)
# print("after gold_data_train.shape:\n",gold_data_train.shape)
# print("after gold_data_test.shape:\n",gold_data_test.shape)
# print("gold_data_train[0]:\n",gold_data_train[0])
# print("gold_data_test[0]:\n",gold_data_test[0])

kosdaq_data_train,kosdaq_data_test, kosdaq_target_train,kosdaq_target_test = train_test_split(
    kosdaq_data, kosdaq_target, train_size=0.9, test_size=0.1, random_state = 44)
# print("after kosdaq_data_train.shape:\n",kosdaq_data_train.shape)
# print("after kosdaq_data_test.shape:\n",kosdaq_data_test.shape)
# print("kosdaq_data_train[0]:\n",kosdaq_data_train[0])
# print("kosdaq_data_test[0]:\n",kosdaq_data_test[0])


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

gold_scaler = StandardScaler()
gold_data_train = scaling3D(gold_data_train, gold_scaler)
gold_data_test = transform3D(gold_data_test, gold_scaler)
print("after scaled gold_data_train.shape",gold_data_train.shape)
print("after scaled gold_data_test.shape:",gold_data_test.shape)
# print("after scaled gold_data_train[0]:",gold_data_train[0])
# print("after scaled gold_data_test[0]:",gold_data_test[0])

kosdaq_scaler = StandardScaler()
kosdaq_data_train = scaling3D(kosdaq_data_train, kosdaq_scaler)
kosdaq_data_test = transform3D(kosdaq_data_test, kosdaq_scaler)
print("after scaled kosdaq_data_train.shape",kosdaq_data_train.shape)
print("after scaled kosdaq_data_test.shape:",kosdaq_data_test.shape)
# print("after scaled kosdaq_data_train[0]:",kosdaq_data_train[0])
# print("after scaled kosdaq_data_test[0]:",kosdaq_data_test[0])


# 1.4 reshape
samsung_data_train = samsung_data_train.reshape(samsung_data_train.shape[0],samsung_data_train.shape[1]*samsung_data_train.shape[2])
samsung_data_test = samsung_data_test.reshape(samsung_data_test.shape[0],samsung_data_test.shape[1]*samsung_data_test.shape[2])
print("after reshape x:", samsung_data_train.shape, samsung_data_test.shape)

bitcom_data_train = bitcom_data_train.reshape(bitcom_data_train.shape[0],bitcom_data_train.shape[1]*bitcom_data_train.shape[2])
bitcom_data_test = bitcom_data_test.reshape(bitcom_data_test.shape[0],bitcom_data_test.shape[1]*bitcom_data_test.shape[2])
print("after reshape x:", bitcom_data_train.shape, bitcom_data_test.shape)

gold_data_train = gold_data_train.reshape(gold_data_train.shape[0],gold_data_train.shape[1]*gold_data_train.shape[2])
gold_data_test = gold_data_test.reshape(gold_data_test.shape[0],gold_data_test.shape[1]*gold_data_test.shape[2])
print("after reshape x:", gold_data_train.shape, gold_data_test.shape)

kosdaq_data_train = kosdaq_data_train.reshape(kosdaq_data_train.shape[0],kosdaq_data_train.shape[1]*kosdaq_data_train.shape[2])
kosdaq_data_test = kosdaq_data_test.reshape(kosdaq_data_test.shape[0],kosdaq_data_test.shape[1]*kosdaq_data_test.shape[2])
print("after reshape x:", kosdaq_data_train.shape, kosdaq_data_test.shape)



modelpath = './model/hcbae22_{epoch:02d}_{val_loss:.4f}.hdf5'
model_save_path = "./save/hcbae22_model.h5"
weights_save_path = './save/hcbae22_weights.h5'




# 2.모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

lstm_num = 64
dense_num1 = 512
dense_num2 = 256
dense_num3 = 128

samsung_model_input = Input(shape=(samsung_data_train.shape[1],1))
dense1 = LSTM(lstm_num, activation='relu', name='dense1_1')(samsung_model_input)
dense1 = Dense(dense_num1, activation='relu', name='dense1_2')(dense1)
dense1 = Dense(dense_num2, activation='relu', name='dense1_3')(dense1)
dense1 = Dense(dense_num3, activation='relu', name='dense1_4')(dense1)
samsung_model_output1 = Dense(10, name='samsung_model_output1')(dense1)

bitcom_model_input = Input(shape=(bitcom_data_train.shape[1],1))
dense2 = LSTM(lstm_num, activation='relu', name='dense2_1')(bitcom_model_input)
dense2 = Dense(dense_num1, activation='relu', name='dense2_2')(dense2)
dense2 = Dense(dense_num2, activation='relu', name='dense2_3')(dense2)
dense2 = Dense(dense_num3, activation='relu', name='dense2_4')(dense2)
bitcom_model_output1 = Dense(10, name='bitcom_model_output1')(dense2)

gold_model_input = Input(shape=(gold_data_train.shape[1],1))
dense3 = LSTM(lstm_num, activation='relu', name='dense3_1')(gold_model_input)
dense3 = Dense(dense_num1, activation='relu', name='dense3_2')(dense3)
dense3 = Dense(dense_num2, activation='relu', name='dense3_3')(dense3)
dense3 = Dense(dense_num3, activation='relu', name='dense3_4')(dense3)
gold_model_output1 = Dense(10, name='gold_model_output1')(dense3)

kosdaq_model_input = Input(shape=(kosdaq_data_train.shape[1],1))
dense4 = LSTM(lstm_num, activation='relu', name='dense4_1')(kosdaq_model_input)
dense4 = Dense(dense_num1, activation='relu', name='dense4_2')(dense4)
dense4 = Dense(dense_num2, activation='relu', name='dense4_3')(dense4)
dense4 = Dense(dense_num3, activation='relu', name='dense4_4')(dense4)
kosdaq_model_output1 = Dense(10, name='kosdaq_model_output1')(dense4)


from tensorflow.keras.layers import concatenate
merge1 = concatenate([samsung_model_output1, bitcom_model_output1,
                       gold_model_output1, kosdaq_model_output1])
middle = Dense(256, name='middle1')(merge1)
middle = Dense(128, name='middle2')(middle)
middle = Dense(64, name='middle3')(middle)

samsung_out = Dense(32, name='output1_1')(middle)
samsung_out = Dense(16, name='output1_2')(samsung_out)
samsung_model_output2 = Dense(1, name='output1_3')(samsung_out)


total_model = Model(inputs=[samsung_model_input,bitcom_model_input,
                            gold_model_input,kosdaq_model_input], 
                    outputs=samsung_model_output2)
total_model.summary()


# 3. 컴파일, 훈련
total_model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    mode='auto',
    verbose=2)

from tensorflow.keras.callbacks import ModelCheckpoint # 모델 체크 포인트
model_check_point = ModelCheckpoint(
    filepath=modelpath,
    monitor='val_loss',
    save_best_only=True,
    mode='auto')

#훈련, 일단 x_train, y_train 입력하고
hist = total_model.fit([samsung_data_train, bitcom_data_train,
                         gold_data_train, kosdaq_data_train], 
                        samsung_target_train,
                        epochs=10000, # 훈련 횟수
                        batch_size=128, # 훈련 데이터단위
                        verbose=1,
                        validation_split=0.2,
                        callbacks=[early_stopping,
                        model_check_point
                        ])

total_model.save(model_save_path)
total_model.save_weights(weights_save_path)

total_model.summary()

# 4. 평가, 예측
result = total_model.evaluate([samsung_data_test, bitcom_data_test,
                                gold_data_test, kosdaq_data_test], 
                                samsung_target_test, batch_size=128)
print("loss: ", result[0])
print("mae: ", result[1])

y_predict = total_model.predict([samsung_data_test, bitcom_data_test,
                                gold_data_test, kosdaq_data_test])
# print("y_predict:", y_predict)



y_recovery = samsung_target_test

# 사용자정의 RMSE 함수
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_recovery, y_predict))


# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y_recovery, y_predict)
print("R2:", r2)



# predict
samsung_pred = samsung_pred.reshape(1, samsung_pred.shape[0],samsung_pred.shape[1])
samsung_pred = transform3D(samsung_pred, samsung_scaler)
samsung_pred = samsung_pred.reshape(1, samsung_pred.shape[1]*samsung_pred.shape[2],1)
print("samsung_pred.shape:",samsung_pred.shape)

bitcom_pred = bitcom_pred.reshape(1, bitcom_pred.shape[0],bitcom_pred.shape[1])
bitcom_pred = transform3D(bitcom_pred, bitcom_scaler)
bitcom_pred = bitcom_pred.reshape(1, bitcom_pred.shape[1]*bitcom_pred.shape[2],1)
print("bitcom_pred.shape:",bitcom_pred.shape)

gold_pred = gold_pred.reshape(1, gold_pred.shape[0],gold_pred.shape[1])
gold_pred = transform3D(gold_pred, gold_scaler)
gold_pred = gold_pred.reshape(1, gold_pred.shape[1]*gold_pred.shape[2],1)
print("gold_pred.shape:",gold_pred.shape)

kosdaq_pred = kosdaq_pred.reshape(1, kosdaq_pred.shape[0],kosdaq_pred.shape[1])
kosdaq_pred = transform3D(kosdaq_pred, kosdaq_scaler)
kosdaq_pred = kosdaq_pred.reshape(1, kosdaq_pred.shape[1]*kosdaq_pred.shape[2],1)
print("kosdaq_pred.shape:",kosdaq_pred.shape)

samsung_predict_today = total_model.predict([samsung_pred, bitcom_pred,
                                            gold_pred, kosdaq_pred])
print("samsung_predict:", int(samsung_predict_today))






# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) # 단위는 찾아보자

plt.subplot(2,1,1) # 2장 중에 첫 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2) # 2장 중에 두 번째
plt.plot(hist.history['mae'], marker='.', c='red')
plt.plot(hist.history['val_mae'], marker='.', c='blue')
plt.grid()
plt.title('mae')
plt.ylabel('mae')
plt.xlabel('epochs')
plt.legend(['mae', 'val_mae'])

plt.show()



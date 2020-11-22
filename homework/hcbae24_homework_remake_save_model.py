import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# 1.데이터
# 1.1 load_data
# 1.2 train_test_split
# 1.3 scaler
# 1.4 reshape
# 2.모델
# 3.컴파일 훈련
# 4.평가 예측



# 데이터는 19일까지
view_size = 10 # 10일씩 자른다

before_days = 2 
# 19일 기준, -2 데이터까지 사용, 즉 17일까지 데이터로 19일 계산
# pred 데이터를 19일까지 사용하면, (주말 제외하고) 23일 데이터를 예상하게 된다

test_rate = 0.2
def split_x2(seq, size):
    bbb = []
    for i in range(len(seq) - size + 1):
        bbb.append(seq[i:(i+size)])
    return np.array(bbb)

# 1. 데이터
# 1.1 load_data
import numpy as np

def load_data(load_data_path):
    temp_data = np.load(load_data_path, allow_pickle=True).astype('float32')
    temp_data = temp_data[:-before_days]
    temp_data = split_x2(temp_data, view_size)
    return temp_data

def load_target(load_target_patt):
    temp_target = np.load(load_target_patt, allow_pickle=True).astype('float32')
    temp_target = temp_target[view_size+before_days-1:]
    return temp_target

samsung_data = load_data('./data/samsung_data.npy')
samsung_target = load_target('./data/samsung_target.npy')
# samsung_data = np.load('./data/samsung_data.npy', allow_pickle=True).astype('float32')
# samsung_data = samsung_data[:-before_days]
# samsung_data = split_x2(samsung_data, view_size)
# print(samsung_data[-before_days:])
# print("samsung_data.shape:",samsung_data.shape)

samsung_target = np.load('./data/samsung_target.npy', allow_pickle=True).astype('float32')
samsung_target = samsung_target[view_size+before_days-1:]
# print(samsung_target[-before_days:])
# print("samsung_target.shape:",samsung_target.shape)

bitcom_data = load_data('./data/bitcom_data.npy')
bitcom_target = load_target('./data/bitcom_target.npy')

gold_data = load_data('./data/gold_data.npy')
gold_target = load_target('./data/gold_target.npy')

kosdaq_data = load_data('./data/kosdaq_data.npy')
kosdaq_target = load_target('./data/kosdaq_target.npy')

print("========== 데이터 로딩 끝 ==========")
# print("samsung_data.shape:", samsung_data.shape)
# print('samsung_target.shape',samsung_target.shape)
# print("bitcom_data.shape:", bitcom_data.shape)
# print('bitcom_target.shape',bitcom_target.shape)
# print("gold_data.shape:", gold_data.shape)
# print('gold_target.shape',gold_target.shape)
# print("kosdaq_data.shape:", kosdaq_data.shape)
# print('kosdaq_target.shape',kosdaq_target.shape)


# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
samsung_data_train,samsung_data_test, samsung_target_train,samsung_target_test = train_test_split(
    samsung_data, samsung_target, train_size=(1.-test_rate), test_size=test_rate, random_state = 44)

bitcom_data_train,bitcom_data_test, bitcom_target_train,bitcom_target_test = train_test_split(
    bitcom_data, bitcom_target, train_size=(1.-test_rate), test_size=test_rate, random_state = 44)

gold_data_train,gold_data_test, gold_target_train,gold_target_test = train_test_split(
    gold_data, gold_target, train_size=(1.-test_rate), test_size=test_rate, random_state = 44)

kosdaq_data_train,kosdaq_data_test, kosdaq_target_train,kosdaq_target_test = train_test_split(
    kosdaq_data, kosdaq_target, train_size=(1.-test_rate), test_size=test_rate, random_state = 44)
print("========== train_test_split 끝 ==========")
# print("after samsung_data_train.shape:\n", samsung_data_train.shape)
# print("after samsung_data_test.shape:\n", bitcom_data_test.shape)


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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
samsung_scaler = StandardScaler()
samsung_data_train = scaling3D(samsung_data_train, samsung_scaler)
samsung_data_test = transform3D(samsung_data_test, samsung_scaler)
# print("after scaled samsung_data_train.shape:",samsung_data_train.shape)
# print("after scaled samsung_data_test.shape:",samsung_data_test.shape)
# print("after scaled samsung_data_train[-1]:",samsung_data_train[-1])
# print("after scaled samsung_data_test[-1]:",samsung_data_test[-1])

bitcom_scaler = StandardScaler()
bitcom_data_train = scaling3D(bitcom_data_train, bitcom_scaler)
bitcom_data_test = transform3D(bitcom_data_test, bitcom_scaler)

gold_scaler = StandardScaler()
gold_data_train = scaling3D(gold_data_train, gold_scaler)
gold_data_test = transform3D(gold_data_test, gold_scaler)

kosdaq_scaler = StandardScaler()
kosdaq_data_train = scaling3D(kosdaq_data_train, kosdaq_scaler)
kosdaq_data_test = transform3D(kosdaq_data_test, kosdaq_scaler)

print("========== scaler 끝 ==========")


# 1.4 reshape
# print("after reshape x:", samsung_data_train.shape, samsung_data_test.shape)
# print("after reshape x:", bitcom_data_train.shape, bitcom_data_test.shape)
# print("after reshape x:", gold_data_train.shape, gold_data_test.shape)
# print("after reshape x:", kosdaq_data_train.shape, kosdaq_data_test.shape)
print("========== reshape 끝 ==========")




# 2.모델
modelpath = './model/hcbae24_rnn_{epoch:02d}_{val_loss:.4f}.hdf5'
model_save_path = "./save/hcbae24_rnn_model.h5"
weights_save_path = './save/hcbae24_rnn_weights.h5'

# model을 채우기 위해 폴더를 비우자
def remove_file(path):
    for file in os.scandir(path):
        os.unlink(file.path)
remove_file('./model')


import datetime
start1 = datetime.datetime.now()

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import GRU, Conv1D
from tensorflow.keras.layers import Dropout

def custom_model(model, node):
    temp = GRU(node, activation='relu')(model)
    temp = Dense(512, activation='relu')(temp)
    temp = Dense(512, activation='relu')(temp)
    temp = Dense(512, activation='relu')(temp)
    temp = Dense(512, activation='relu')(temp)
    temp = Dropout(0.2)(temp)
    return temp
    
samsung_model_input = Input(shape=(samsung_data_train.shape[1],samsung_data_train.shape[2]))
samsung_model_output1 = custom_model(samsung_model_input, 64)

bitcom_model_input = Input(shape=(bitcom_data_train.shape[1],bitcom_data_train.shape[2]))
bitcom_model_output1 = custom_model(bitcom_model_input, 1)

gold_model_input = Input(shape=(gold_data_train.shape[1],gold_data_train.shape[2]))
gold_model_outout1 = custom_model(gold_model_input, 1)

kosdaq_model_input = Input(shape=(kosdaq_data_train.shape[1],kosdaq_data_train.shape[2]))
kosdaq_model_outout1 = custom_model(kosdaq_model_input, 1)


from tensorflow.keras.layers import concatenate
samsung_out = concatenate([samsung_model_output1, bitcom_model_output1,
                       gold_model_outout1, kosdaq_model_outout1])
samsung_out = Dense(512, activation='relu')(samsung_out)
samsung_out = Dense(512, activation='relu')(samsung_out)
samsung_out = Dense(512, activation='relu')(samsung_out)
samsung_out = Dense(512, activation='relu')(samsung_out)
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
    patience=100,
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
                        batch_size=512, # 훈련 데이터단위
                        verbose=1,
                        validation_split=0.25,
                        callbacks=[early_stopping,
                        model_check_point
                        ])

total_model.save(model_save_path)
total_model.save_weights(weights_save_path)


# 4. 평가, 예측
result = total_model.evaluate([samsung_data_test, bitcom_data_test,
                                gold_data_test, kosdaq_data_test], 
                                samsung_target_test, batch_size=512)
# print("loss: ", result[0])
# print("mae: ", result[1])
y_predict = total_model.predict([samsung_data_test, bitcom_data_test,
                                gold_data_test, kosdaq_data_test])
# print("y_predict:", y_predict)

y_recovery = samsung_target_test
# print("y_test:", y_recovery)
# print("y_predict:", y_predict)

# 사용자정의 RMSE 함수
from sklearn.metrics import mean_squared_error, mean_squared_log_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_recovery, y_predict))

# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y_recovery, y_predict)
print("R2:", r2)
print("loss: ", result[0])
print("mae: ", result[1])
print("RMSLE:",np.sqrt(mean_squared_log_error(y_recovery, y_predict)))

end1 = datetime.datetime.now()
print('model load~evluate~predict total time:', (end1-start1))


# pred
pred_size = 100
def make_pred(path, scaler):
    data = np.load(path, allow_pickle=True).astype('float32')
    data = split_x2(data, view_size)
    data = data[(data.shape[0]-samsung_data_train.shape[0]):]
    data = transform3D(data, scaler)
    data = data[-pred_size:]
    return data

# samsung_data_pred = np.load('./data/samsung_data.npy', allow_pickle=True).astype('float32')
# samsung_data_pred = split_x2(samsung_data_pred, pred_size)
# samsung_data_pred = samsung_data_pred[(samsung_data_pred.shape[0]-samsung_data_train.shape[0]):]
# samsung_data_pred = transform3D(samsung_data_pred, samsung_scaler)
samsung_data_pred = make_pred('./data/samsung_data.npy', samsung_scaler)
# print("samsung_data_pred.shape[-1]:",samsung_data_pred[-1])
# print("samsung_data_pred.shape:",samsung_data_pred.shape)

bitcom_data_pred = make_pred('./data/bitcom_data.npy', bitcom_scaler)
gold_data_pred = make_pred('./data/gold_data.npy', gold_scaler)
kosdaq_data_pred = make_pred('./data/kosdaq_data.npy', kosdaq_scaler)


samsung_predict_price = total_model.predict([samsung_data_pred, bitcom_data_pred,
                                            gold_data_pred, kosdaq_data_pred])
print("samsung_predict_price[-5:]:\r\n", samsung_predict_price[-5:])
print("samsung_predict_price.shape:",samsung_predict_price.shape)

samsung_target_pred = np.load('./data/samsung_target.npy', allow_pickle=True).astype('float32')
samsung_target_pred = samsung_target_pred[-(pred_size-before_days):]
print("samsung_target_pred.shape:",samsung_target_pred.shape)


# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) # 단위는 찾아보자
plt.plot(samsung_predict_price, marker='.', c='red', label='samsung_predict_cost')
plt.plot(samsung_target_pred, marker='.', c='blue', label='samsung_target_pred')
plt.grid()
plt.ylabel('samsung_predict_cost')
plt.legend(loc='upper right')
plt.show()


'''
plt.subplot(2,1,1) # 2장 중에 첫 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.yscale('log')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2) # 2장 중에 두 번째
plt.plot(hist.history['mae'], marker='.', c='red')
plt.plot(hist.history['val_mae'], marker='.', c='blue')
plt.grid()
plt.title('mae')
plt.ylabel('mae')
plt.yscale('log')
plt.xlabel('epochs')
plt.legend(['mae', 'val_mae'])

plt.show()

'''


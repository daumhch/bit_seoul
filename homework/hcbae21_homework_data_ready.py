import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


pick_name = '시가'


import numpy as np
import pandas as pd


samsung = pd.read_csv('./data/csv/삼성전자 1120.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=',' # 구분 기호
                        )
bitcom = pd.read_csv('./data/csv/비트컴퓨터 1120.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=',' # 구분 기호
                        )
gold = pd.read_csv('./data/csv/금현물.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=',' # 구분 기호
                        )
kosdaq= pd.read_csv('./data/csv/코스닥.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=',' # 구분 기호
                        )
print("=========== original data ===========")
# print(type(samsung))
# print(samsung.shape)
# print(samsung.head())
# print(type(bitcom))
# print(bitcom.shape)
# print(bitcom.head())
# print(type(gold))
# print(gold.shape)
# print(gold.head())
# print(type(kosdaq))
# print(kosdaq.shape)
# print(kosdaq.head())

samsung = samsung[['시가','종가','저가','고가', '거래량', '기관', '외인(수량)']]
samsung = samsung.iloc[1:626]

bitcom = bitcom[['시가','종가','저가','고가', '금액(백만)', '기관']]
bitcom = bitcom.iloc[1:626]

gold = gold[['시가','종가','저가','고가', '거래대금(백만)']]
gold = gold.iloc[1:626]

kosdaq = kosdaq[['시가','고가','저가']]
kosdaq = kosdaq.iloc[1:626]

print("=========== after cutting ===========")
# print(type(samsung))
# print(samsung.shape)
# print(samsung.head())
# print(type(bitcom))
# print(bitcom.shape)
# print(bitcom.head())
# print(type(gold))
# print(gold.shape)
# print(gold.head())
# print(type(kosdaq))
# print(kosdaq.shape)
# print(kosdaq.head())

samsung = samsung.sort_values(['일자'],ascending=['True'])
bitcom = bitcom.sort_values(['일자'],ascending=['True'])
gold = gold.sort_values(['일자'],ascending=['True'])
kosdaq = kosdaq.sort_values(['일자'],ascending=['True'])

# 콤마 제거 후 문자를 정수로 변환
def eraseComma(data):
    for i in range(len(data.index)):
        for j in range(len(data.iloc[i])):
            if isinstance(data.iloc[i,j], str):
                data.iloc[i,j]=int(data.iloc[i,j].replace(',',''))
            # print("i,j:",i,"/",j,"/",data.iloc[i,j])
    return data
samsung = eraseComma(samsung)
bitcom = eraseComma(bitcom)
gold = eraseComma(gold)
kosdaq = eraseComma(kosdaq)

print("========== after sort ==========")
# print(samsung.head())
# print(samsung.tail())
# print('samsung.shape',samsung.shape)
# print(bitcom.head())
# print(bitcom.tail())
# print('bitcom.shape',bitcom.shape)
# print(gold.head())
# print(gold.tail())
# print('gold.shape',gold.shape)
# print(kosdaq.head())
# print(kosdaq.tail())
# print('kosdaq.shape',kosdaq.shape)



print("========== y numpy 저장 ==========")
samsung_target = samsung[pick_name]
# print(samsung_target.head())
# print(samsung_target.tail())
# print('samsung_target.shape',samsung_target.shape)
samsung_target = samsung_target.to_numpy()
np.save('./data/samsung_target.npy',arr=samsung_target)

bitcom_target = bitcom[pick_name]
# print(bitcom_target.head())
# print(bitcom_target.tail())
# print('bitcom_target.shape',bitcom_target.shape)
bitcom_target = bitcom_target.to_numpy()
np.save('./data/bitcom_target.npy',arr=bitcom_target)

gold_target = bitcom[pick_name]
# print(gold_target.head())
# print(gold_target.tail())
# print('gold_target.shape',gold_target.shape)
gold_target = gold_target.to_numpy()
np.save('./data/gold_target.npy',arr=gold_target)

kosdaq_target = bitcom[pick_name]
# print(kosdaq_target.head())
# print(kosdaq_target.tail())
# print('kosdaq_target.shape',kosdaq_target.shape)
kosdaq_target = kosdaq_target.to_numpy()
np.save('./data/kosdaq_target.npy',arr=kosdaq_target)



print("========== x numpy 저장 ==========")
samsung_data = samsung.to_numpy()
np.save('./data/samsung_data.npy',arr=samsung_data)

bitcom_data = bitcom.to_numpy()
np.save('./data/bitcom_data.npy',arr=bitcom_data)

gold_data = gold.to_numpy()
np.save('./data/gold_data.npy',arr=gold_data)

kosdaq_data = kosdaq.to_numpy()
np.save('./data/kosdaq_data.npy',arr=kosdaq_data)





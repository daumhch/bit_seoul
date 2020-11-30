import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np
import pandas as pd
from datetime import datetime

merge_csv = pd.read_csv('./project2/merge_csv.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
print("merge_csv.shape:",merge_csv.shape)

merge_csv['Date'] = pd.to_datetime(merge_csv['Date'])


print("merge_csv.head():\r\n",merge_csv.head())
print("merge_csv.shape:",merge_csv.shape)

internal_validity = merge_csv[merge_csv['Date']<datetime(2014,1,1)]
print("internal_validity.shape:", internal_validity.shape)

external_validity = merge_csv[merge_csv['Date']>=datetime(2014,1,1)]
print("external_validity.shape:", external_validity.shape)


# 데이터 병합하기 위해 사용한 Date와 Time은
# 삭제하여 분류 모델에는 사용하지 않는다
internal_validity = internal_validity.drop(['Date'], axis=1)
internal_validity = internal_validity.drop(['Time'], axis=1)
internal_validity = internal_validity.drop(['Local_Authority_(Highway)'], axis=1)
internal_validity = internal_validity.drop(['LSOA_of_Accident_Location'], axis=1)

external_validity = external_validity.drop(['Date'], axis=1)
external_validity = external_validity.drop(['Time'], axis=1)
external_validity = external_validity.drop(['Local_Authority_(Highway)'], axis=1)
external_validity = external_validity.drop(['LSOA_of_Accident_Location'], axis=1)

print("internal_validity:\r\n",internal_validity)
print("external_validity:\r\n",external_validity)

csv_index = internal_validity.columns
np.save('./project2/csv_index.npy',arr=csv_index)

iv_target_npy = internal_validity['Casualty_Severity'].to_numpy()
print("iv_target.shape:",iv_target_npy.shape)
np.save('./project2/iv_target.npy',arr=iv_target_npy)

iv_data_npy = internal_validity.drop(['Casualty_Severity'], axis=1).to_numpy()
print("iv_data.shape:",iv_data_npy.shape)
np.save('./project2/iv_data.npy',arr=iv_data_npy)


ev_target_npy = external_validity['Casualty_Severity'].to_numpy()
print("ev_target.shape:",ev_target_npy.shape)
np.save('./project2/ev_target.npy',arr=ev_target_npy)

ev_data_npy = external_validity.drop(['Casualty_Severity'], axis=1).to_numpy()
print("ev_data.shape:",ev_data_npy.shape)
np.save('./project2/ev_data.npy',arr=ev_data_npy)


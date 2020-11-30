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


# 데이터 병합하기 위해 사용한 Date와 Time은
# 삭제하여 분류 모델에는 사용하지 않는다
merge_csv = merge_csv.drop(['Date'], axis=1)
merge_csv = merge_csv.drop(['Time'], axis=1)
merge_csv = merge_csv.drop(['Local_Authority_(Highway)'], axis=1)
merge_csv = merge_csv.drop(['LSOA_of_Accident_Location'], axis=1)


print("merge_csv:\r\n",merge_csv)

csv_index = merge_csv.columns
np.save('./project2/csv_index.npy',arr=csv_index)

merge_target_npy = merge_csv['Casualty_Severity'].to_numpy()
print("merge_npy.shape:",merge_target_npy.shape)
np.save('./project2/merge_target.npy',arr=merge_target_npy)

merge_data_npy = merge_csv.drop(['Casualty_Severity'], axis=1).to_numpy()
print("merge_data.shape:",merge_data_npy.shape)
np.save('./project2/merge_data.npy',arr=merge_data_npy)



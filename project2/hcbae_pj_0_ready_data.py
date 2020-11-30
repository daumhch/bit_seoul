import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

import numpy as np
import pandas as pd


accidents = pd.read_csv('./project2/Accidents0514.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
print("accidents.shape:",accidents.shape)


vehicles = pd.read_csv('./project2/Vehicles0514.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
print("vehicles.shape:",vehicles.shape)


casualties = pd.read_csv('./project2/Casualties0514.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
print("casualties.shape:",casualties.shape)

ftse = pd.read_csv('./project2/Index2018.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
print("ftse.shape:",ftse.shape)


merge_csv = pd.merge(accidents, vehicles, how='left', on='Accident_Index')
merge_csv = pd.merge(merge_csv, casualties, how='left', on='Accident_Index')
merge_csv = pd.merge(merge_csv, ftse, how='left', on='Date')


# ============ drop -1 시작 ============
def deleteValue(arr, value):
    for cnt in range(0, arr.shape[1]):
        arr = arr.drop(arr[arr.iloc[:,cnt]== value].index)
    return arr
print("before erase -1 / merge_csv.shape:",merge_csv.shape)
merge_csv = deleteValue(merge_csv, -1)
print("after erase -1 / merge_csv.shape:",merge_csv.shape)
# ============ drop -1 끝 ============


# ============ drop nan 시작 ============
print("before erase NAN merge_csv.shape:",merge_csv.shape)
merge_csv = merge_csv.dropna(axis=0)
print("after erase NAN merge_csv.shape:",merge_csv.shape)
# ============ drop nan 끝 ============


print(merge_csv.head())
print(merge_csv.tail())
print("merge_csv.shape:",merge_csv.shape)
print("merge_csv.shape:",merge_csv.columns)


# 데이터 병합하기 위해 사용한 Accident_Index는
# 삭제하여 분류 모델에는 사용하지 않는다
merge_csv = merge_csv.drop(['Accident_Index'], axis=1)



merge_csv.to_csv('./project2/merge_csv.csv')

# merge_csv_index = merge_csv.columns
# np.save('./project2/merge_index.npy',arr=merge_csv_index)

# merge_target_npy = merge_csv['Casualty_Severity'].to_numpy()
# print("merge_target_npy.shape:",merge_target_npy.shape)
# np.save('./project2/merge_target.npy',arr=merge_target_npy)

# merge_data_npy = merge_csv.drop(['Casualty_Severity'], axis=1).to_numpy()
# print("merge_data_npy.shape:",merge_data_npy.shape)
# np.save('./project2/merge_data.npy',arr=merge_data_npy)





import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# 1.데이터
#https://www.kaggle.com/benoit72/uk-accidents-10-years-history-with-many-variables
import numpy as np
import pandas as pd

accidents = pd.read_csv('./project/Accidents0514.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
# 열 삭제
# accidents = accidents.drop(["Location_Easting_OSGR", "Location_Northing_OSGR", "Longitude", "Latitude"], axis=1)

accidents = accidents[[
    'Accident_Index',
    'Date',
    'Accident_Severity',
    'Number_of_Vehicles',
    'Number_of_Casualties',
    'Road_Type',
    'Speed_limit',
    'Junction_Detail',
    'Junction_Control',
    'Light_Conditions',
    'Weather_Conditions',
    'Road_Surface_Conditions'
    ]]
# print(accidents.head())
print("accidents.shape:",accidents.shape)


vehicles = pd.read_csv('./project/Vehicles0514.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
vehicles = vehicles[[
    'Accident_Index',
    'Vehicle_Type',
    'Vehicle_Manoeuvre',
    'Junction_Location',
    'Skidding_and_Overturning',
    '1st_Point_of_Impact',
    'Sex_of_Driver',
    'Age_of_Driver',
    'Age_Band_of_Driver',
    'Engine_Capacity_(CC)',
    'Age_of_Vehicle'
    ]]
# print(vehicles.head())
print("vehicles.shape:",vehicles.shape)


casualties = pd.read_csv('./project/Casualties0514.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
casualties = casualties[[
    'Accident_Index',
    'Casualty_Class',
    'Sex_of_Casualty',
    'Age_of_Casualty',
    'Age_Band_of_Casualty',
    'Casualty_Severity',
    'Car_Passenger',
    'Casualty_Type'
    ]]
# print(casualties.head())
print("casualties.shape:",casualties.shape)


# https://www.kaggle.com/regaipkurt/financial-markets
ftse = pd.read_csv('./project/Index2018.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
ftse = ftse[[
    'Date',
    'ftse'
    ]]
# print(ftse.head())
print("ftse.shape:",ftse.shape)


import datetime as dt
merge_csv = pd.merge(accidents, vehicles, how='left', on='Accident_Index')
merge_csv = pd.merge(merge_csv, casualties, how='left', on='Accident_Index')
merge_csv = pd.merge(merge_csv, ftse, how='left', on='Date')
# merge_csv['Date'] = pd.to_datetime(merge_csv['Date']).astype('int64')
# print(type(merge_csv['Date'][0]))
# Date를 int64로 변환하여 사용할 수 있지만,
# Date의 증가하는 특성과, 사고 정도 '분류'는 상관관계가 없다고 판단하여,
# 데이터 병합하기 위해 사용한 Accident_Index와 Date 컬럼은
# 이 이후 삭제하여 분류 모델에는 사용하지 않는다
merge_csv = merge_csv.drop(['Accident_Index','Date'], axis=1)
print(merge_csv.head())
print(merge_csv.tail())
print("merge_csv.shape:",merge_csv.shape)
print("merge_csv.shape:",merge_csv.columns)







'''
merge_csv_index = merge_csv.columns
np.save('./project/merge_index.npy',arr=merge_csv_index)

merge_target_npy = merge_csv['Casualty_Severity'].to_numpy()
print("merge_target_npy.shape:",merge_target_npy.shape)
np.save('./project/merge_target.npy',arr=merge_target_npy)

merge_data_npy = merge_csv.drop(['Casualty_Severity'], axis=1).to_numpy()
print("merge_data_npy.shape:",merge_data_npy.shape)
np.save('./project/merge_data.npy',arr=merge_data_npy)
'''

'''
# load_npy = np.load('./project/merge_npy.npy', allow_pickle=True)
# print(load_npy[:5])


merge_csv = pd.read_csv('./project/merge_csv.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )

print(merge_csv.head())
print("merge_csv.shape:",merge_csv.shape)
'''






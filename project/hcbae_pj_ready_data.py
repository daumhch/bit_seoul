import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# 1.데이터
#https://www.kaggle.com/benoit72/uk-accidents-10-years-history-with-many-variables
import numpy as np
import pandas as pd

'''
accidents = pd.read_csv('./project/Accidents0514.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
# 열 삭제
# accidents = accidents.drop(["Location_Easting_OSGR", "Location_Northing_OSGR", "Longitude", "Latitude"], axis=1)

accidents = accidents[[
    'Accident_Index',
    'Accident_Severity',
    'Number_of_Vehicles',
    'Number_of_Casualties',
    'Time',
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


merge_csv = pd.merge(accidents, vehicles, how='outer', on='Accident_Index')
merge_csv = pd.merge(merge_csv, casualties, how='outer', on='Accident_Index')


print(merge_csv.head())
print("merge_csv.shape:",merge_csv.shape)
'''



'''
merge_csv.to_csv('./project/merge_csv.csv')

merge_npy = merge_csv.to_numpy()
print("merge_npy.shape:",merge_npy.shape)
np.save('./project/merge_npy.npy',arr=merge_npy)
'''
# load_npy = np.load('./project/merge_npy.npy', allow_pickle=True)
# print(load_npy[:5])


merge_csv = pd.read_csv('./project/merge_csv.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )

# 시간을 분으로 저장하자 (숫자 저장하자)
print(merge_csv.head())
print("merge_csv.shape:",merge_csv.shape)







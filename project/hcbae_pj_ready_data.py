import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



# 1.데이터
#https://www.kaggle.com/benoit72/uk-accidents-10-years-history-with-many-variables
import numpy as np
import pandas as pd

datasets = pd.read_csv('./project/Accidents0514.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
# 열 삭제
datasets = datasets.drop(["Location_Easting_OSGR", "Location_Northing_OSGR", "Longitude", "Latitude"], axis=1)

datasets = datasets[['Accident_Index','Police_Force','Accident_Severity']]
print(datasets.head())
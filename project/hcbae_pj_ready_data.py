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
accidents = accidents.drop(["Location_Easting_OSGR", "Location_Northing_OSGR", "Longitude", "Latitude"], axis=1)

accidents = accidents[['Accident_Index','Police_Force','Accident_Severity']]
print(accidents.head())

casualties = pd.read_csv('./project/Casualties0514.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )
# casualties = casualties[['Casualty_Reference','Casualty_Class','Casualty_Type']]
print(casualties.head())

count_data = casualties.groupby('Casualty_Reference')['Casualty_Reference'].count()
print(count_data[:20])
import matplotlib.pyplot as plt
count_data.plot()
plt.show()







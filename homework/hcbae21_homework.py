import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



import numpy as np
import pandas as pd


datasets = pd.read_csv('./data/csv/삼성전자 1120.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=',' # 구분 기호
                        )
print(datasets.shape)
print(datasets.head())
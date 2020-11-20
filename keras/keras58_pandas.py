import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np
import pandas as pd


'''
datasets = pd.read_csv('./data/csv/iris_ys.csv', 
                        header=None, # 첫 번 째 행 = 헤더다
                        index_col=None, # 컬럼 시작 번호
                        sep=',' # 구분 기호
                        )

print(datasets) # header와 index_col 설정에 따라 아래처럼도 출력된다
#          0             1            2             3            4        5
# 0      NaN  sepal_length  sepal_width  petal_length  petal_width  species
# 1      1.0           5.1          3.5           1.4          0.2        0
# 2      2.0           4.9            3           1.4          0.2        0
# 3      3.0           4.7          3.2           1.3          0.2        0
# 4      4.0           4.6          3.1           1.5          0.2        0
# ..     ...           ...          ...           ...          ...      ...
# 146  146.0           6.7            3           5.2          2.3        2
# 147  147.0           6.3          2.5             5          1.9        2
# 148  148.0           6.5            3           5.2            2        2
# 149  149.0           6.2          3.4           5.4          2.3        2
# 150  150.0           5.9            3           5.1          1.8        2
'''



datasets = pd.read_csv('./data/csv/iris_ys.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        sep=',' # 구분 기호
                        )

#      sepal_length  sepal_width  petal_length  petal_width  species
# 1             5.1          3.5           1.4          0.2        0
# 2             4.9          3.0           1.4          0.2        0
# 3             4.7          3.2           1.3          0.2        0
# 4             4.6          3.1           1.5          0.2        0
# 5             5.0          3.6           1.4          0.2        0
# ..            ...          ...           ...          ...      ...
# 146           6.7          3.0           5.2          2.3        2
# 147           6.3          2.5           5.0          1.9        2
# 148           6.5          3.0           5.2          2.0        2
# 149           6.2          3.4           5.4          2.3        2
# 150           5.9          3.0           5.1          1.8        2

# print(datasets) # 출력에서 헤더가 보이는 건, 사용자 편의다

print(datasets.shape) # (150, 5)

# header = None, 0, 1  / index_col = None, 0, 1일 때
# 각각의 상황에서 어떻게 출력되는지 꼭 확인하라
'''
datasets = pd.read_csv('./data/csv/iris_ys.csv', header=None, index_col=None, sep=',')
print("header=None, index_col=None", datasets.shape)
print("header=None, index_col=None datasets\n",datasets)
datasets = pd.read_csv('./data/csv/iris_ys.csv', header=None, index_col=0, sep=',')
print("header=None, index_col=0", datasets.shape)
print("header=None, index_col=0 datasets\n",datasets)
datasets = pd.read_csv('./data/csv/iris_ys.csv', header=None, index_col=1, sep=',')
print("header=None, index_col=1", datasets.shape)
print("header=None, index_col=1 datasets\n",datasets)

datasets = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=None, sep=',')
print("header=0, index_col=None", datasets.shape)
print("header=0, index_col=None datasets\n",datasets)
datasets = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0, sep=',')
print("header=0, index_col=0", datasets.shape)
print("header=0, index_col=0 datasets\n",datasets)
datasets = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=1, sep=',')
print("header=0, index_col=1", datasets.shape)
print("header=0, index_col=1 datasets\n",datasets)

datasets = pd.read_csv('./data/csv/iris_ys.csv', header=1, index_col=None, sep=',')
print("header=1, index_col=None", datasets.shape)
print("header=1, index_col=None datasets\n",datasets)
datasets = pd.read_csv('./data/csv/iris_ys.csv', header=1, index_col=0, sep=',')
print("header=1, index_col=0", datasets.shape)
print("header=1, index_col=0 datasets\n",datasets)
datasets = pd.read_csv('./data/csv/iris_ys.csv', header=1, index_col=1, sep=',')
print("header=1, index_col=1", datasets.shape)
print("header=1, index_col=1 datasets\n",datasets)

# header=None, index_col=None (151, 6)
# header=None, index_col=0    (151, 5)
# header=None, index_col=1    (151, 5)
# header=0,    index_col=None (150, 6)
# header=0,    index_col=0    (150, 5)
# header=0,    index_col=1    (150, 5)
# header=1,    index_col=None (149, 6)
# header=1,    index_col=0    (149, 5)
# header=1,    index_col=1    (149, 5)

# (150, 5)는 header=0, index_col=0 또는 1, 일 때 이다

# header=0, index_col=1도 (150, 5)이지만,
# 결과에서는 header가 하나씩 밀려서 인식한다, 즉 옳은 방식이 아니다

'''

print(datasets.head()) # 위에서부터 5개만 보여줌
print(datasets.tail()) # 아래에서부터 5개만 보여줌
print(type(datasets))

aaa = datasets.to_numpy()
# datasets_numpy = datasets.values
print(type(aaa))
print(aaa.shape)

np.save('./data/iris_ys_pd.npy',arr=aaa)







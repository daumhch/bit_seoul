import pandas as pd
import numpy as np

from numpy.random import randn
np.random.seed(100)

data = randn(5,4)
print(data)

df = pd.DataFrame(data, 
                    index='A B C D E'.split(),
                    columns='가 나 다 라'.split())
print(df)


data2 = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16],
    [17,18,19,20]]
# 5행 4열
df2 = pd.DataFrame(data2,
                    index=['A', 'B', 'C', 'D', 'E'],
                    columns=['가','나','다','라'])
print(df2)
#    가   나   다   라
# A   1   2   3   4
# B   5   6   7   8
# C   9  10  11  12
# D  13  14  15  16
# E  17  18  19  20


df3 = pd.DataFrame(np.array([[1,2,3],[4,5,6]]))
print(df3) # 2행 3열
# 판다스는 인덱스를 자동으로 넣어준다


# column
print("df2['나']: \n", df2['나'])
# df2['나']:
# A     2
# B     6
# C    10
# D    14
# E    18
# Name: 나, dtype: int64
# 컬럼으로도 출력이 가능하다


print("df2['나','라']: \n", df2[['나','라']])
# df2['나','라']:
#      나   라
# A   2   4
# B   6   8
# C  10  12
# D  14  16
# E  18  20
# 컬럼을 선택해서 출력이 가능하다


# print("df2[0]: \n", df2[0])
# 컬럼명으로 해야 에러가 나지 않는다



# 판다스에는 loc iloc가 있다 ### 굉장히 중요하다

# loc = location
# print("df2.loc['나']:\n", df2.loc['나'])
# loc에는 컬럼명이 들어가면 안 된다 에러가 난다
# loc는 행에서 먼저 사용해야 한다

# iloc = index location
print("df2.iloc[:,2]:\n", df2.iloc[:,2])
# df2.iloc[:,2]:
# A     3
# B     7
# C    11
# D    15
# E    19
# Name: 다, dtype: int64



# print("df2[:,2]:\n", df2[:,2])
# 이건 에러난다
# 판다스에서는 컬럼명으로 접근하거나, iloc로 접근할 수 있다



# row(열)
print("df2.loc['A']:\n", df2.loc['A'])
# df2.loc['A']:
# 가    1
# 나    2
# 다    3
# 라    4
# Name: A, dtype: int64

print("df2.loc[['A','C']]:\n", df2.loc[['A','C']])
#     가   나   다   라
# A  1   2   3   4
# C  9  10  11  12


print("df2.iloc[0]:\n", df2.iloc[0])
print("df2.iloc[[0,2]]:\n", df2.iloc[[0,2]])
# df2.iloc[0]:
# 가    1
# 나    2
# 다    3
# 라    4
# Name: A, dtype: int64
# df2.iloc[[0,2]]:
#     가   나   다   라
# A  1   2   3   4
# C  9  10  11  12



# 행렬
print("df2.loc[['A','B'],['나','다']]:\n", df2.loc[['A','B'],['나','다']])
# df2.loc[['A','B'],['나','다']]:
#     나  다
# A  2  3
# B  6  7


# 한 개의 값만 확인 [인덱스, 컬럼명]
print("df2.loc['E','다']:\n",df2.loc['E','다'])
# df2.loc['E','다']:
# 19

print("df2.iloc[4,2]:\n",df2.iloc[4,2])
# df2.iloc[4,2]:
# 19

print("df2.iloc[4][2]:\n",df2.iloc[4][2])
# df2.iloc[4][2]:
# 19



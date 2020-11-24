import pandas as pd

wine = pd.read_csv('./data/csv/winequality-white.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=';' # 구분 기호
                        )
print(wine)
count_data = wine.groupby('quality')['quality'].count()
print(count_data)
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5
# Name: quality, dtype: int64

import matplotlib.pyplot as plt
count_data.plot()
plt.show()







# winequality-white.csv

# 1.데이터
# 1.1 load_data
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

import pandas as pd

wine = pd.read_csv('./data/csv/winequality-white.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=';' # 구분 기호
                        )
wine = wine.to_numpy()
x = wine[:,:-1]
y = wine[:,-1]


x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print("x_train.shape",x_train.shape)
print("x_test.shape",x_test.shape)


# 2.모델
# model = SVC()
# model = LinearSVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
model = RandomForestClassifier()
# model = RandomForestRegressor()


# 3.훈련
model.fit(x_train, y_train)


# 4.평가 예측
score = model.score(x_test, y_test)
print("model.score:",score)
# 분류에서는 accuracy_score를 추가 한다
# 회귀에서는 r2_score를 추가 한다

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score:',acc)

r2 = r2_score(y_test, y_predict)
print('r2_score:',r2)

print(y_test[:10])
print(y_predict[:10])





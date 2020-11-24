# 4개의 모델을 완성하시오



# 1.데이터
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)
x = iris.iloc[:,:-1]
y = iris.iloc[:,-1]
print(x.shape)
print(y.shape)

x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print("x_train.shape",x_train.shape)
print("x_test.shape",x_test.shape)



# 2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

model = SVC()
model2 = LinearSVC()
model3 = KNeighborsClassifier()
model4 = RandomForestClassifier()



# 3. 훈련
model.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(model, x_train, y_train, cv=kfold)
scores2 = cross_val_score(model2, x_train, y_train, cv=kfold)
scores3 = cross_val_score(model3, x_train, y_train, cv=kfold)
scores4 = cross_val_score(model4, x_train, y_train, cv=kfold)

print("SVC scores:                    ",scores,"/",sum(scores))
print("LinearSVC scores2:             ",scores2,"/",sum(scores2))
print("KNeighborsClassifier scores3:  ",scores3,"/",sum(scores3))
print("RandomForestClassifier scores4:",scores4,"/",sum(scores4))



'''
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
'''







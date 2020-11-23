import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

# 1.데이터
x, y = load_iris(return_X_y=True)

x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, train_size=0.8)

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



'''
모델별로 결과에 대해 정리

- SVC
model.score: 0.9666666666666667
accuracy_score: 0.9666666666666667
r2_score: 0.9516908212560387
[2 0 1 1 2 0 2 2 2 1]
[2 0 1 1 2 0 2 2 2 1]


- LinearSVC
model.score: 0.9333333333333333
accuracy_score: 0.9333333333333333
r2_score: 0.9033816425120773
[2 0 1 1 2 0 2 2 2 1]
[2 0 1 1 2 0 2 2 2 1]


- KNeighborsClassifier
model.score: 0.9333333333333333
accuracy_score: 0.9333333333333333
r2_score: 0.9033816425120773
[2 0 1 1 2 0 2 2 2 1]
[2 0 1 1 2 0 2 2 2 1]


- KNeighborsRegressor
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets


- RandomForestClassifier
model.score: 0.9666666666666667
accuracy_score: 0.9666666666666667
r2_score: 0.9516908212560387
[2 0 1 1 2 0 2 2 2 1]
[2 0 1 1 2 0 2 2 2 1]


- RandomForestRegressor
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets

'''




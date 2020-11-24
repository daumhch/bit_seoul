# 유방암 데이터
# 모델은 RandomForestClassifier

# 이렇게 돌리면 2+4+4++41 번 검증한다
# parameters = [
# {'n_estimators':[100,200,300,400]}, # 100,200
# {'max_depth':[6,8,10,12]}, # 6,8,10,12
# {'min_samples_leaf':[3,5,7,10]}, # 3,5,7,10
# {'min_samples_split':[2,3,5,10]}, # 2,3,5,10
# {'n_jobs':[-1]} # -1
# ]

# 이렇게 돌리면 2x4x4x4x1 번 검증한다
# parameters = [
#     {'n_estimators':[100,200], # 100,200
#     'max_depth':[6,8,10,12], # 6,8,10,12
#     'min_samples_leaf':[3,5,7,10], # 3,5,7,10
#     'min_samples_split':[2,3,5,10], # 2,3,5,10
#     'n_jobs':[-1]} # -1
# ]




import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
x, y = load_breast_cancer(return_X_y=True)

parameters = [
    {'n_estimators':[100,200,300,400], # 100,200
    'max_depth': np.array(range(3,15)), # 6,8,10,12
    'min_samples_leaf': np.array(range(3,15)), # 3,5,7,10
    'min_samples_split': np.array(range(3,15)), # 2,3,5,10
    'n_jobs':[-1]} # -1
]

# 1.2 train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


# 2. 모델
kfold = KFold(n_splits=5, shuffle=True)

model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1) # kfold가 5번 x 20번 = 총 100번


# 3. 훈련
model.fit(x_train,y_train)


# 4. 평가 예측
print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)
# 최적의 매개변수: RandomForestClassifier(max_depth=10, min_samples_leaf=3, min_samples_split=5,
#                        n_jobs=-1)
# 최적의 파라미터: {'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 100, 'n_jobs': -1}
# 최종정답률: 0.9649122807017544

y_predict = model.predict(x_test)
print('최종정답률:',accuracy_score(y_test, y_predict))




# 모델은 RandomForest

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

from sklearn.datasets import load_boston
x, y = load_boston(return_X_y=True)

parameters = [
    {'n_estimators':[100,200], # 100,200
    'max_depth': [6,8,10,12], # 6,8,10,12
    'min_samples_leaf': [3,5,7,10], # 3,5,7,10
    'min_samples_split': [2,3,5,10], # 2,3,5,10
    'n_jobs':[-1]} # -1
]

# 1.2 train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


# 2. 모델
kfold = KFold(n_splits=5, shuffle=True)

model = GridSearchCV(RandomForestRegressor(), 
                    parameters, 
                    cv=kfold, 
                    verbose=2) # kfold가 5번 x 20번 = 총 100번


# 3. 훈련
model.fit(x_train,y_train)


# 4. 평가 예측
print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)


y_predict = model.predict(x_test)
print('최종정답률:',r2_score(y_test, y_predict))

# 최적의 매개변수: RandomForestRegressor(max_depth=8, min_samples_leaf=3, min_samples_split=3,
#                       n_estimators=200, n_jobs=-1)
# 최적의 파라미터: {'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 
# 3, 'n_estimators': 200, 'n_jobs': -1}
# 최종정답률: 0.8808311258077494


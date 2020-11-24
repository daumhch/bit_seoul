# 1.데이터
# 1.1 load_data
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston
x, y = load_boston(return_X_y=True)

# 1.2 train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


parameters = [
    {'anyway__n_estimators':np.array(range(100,1000,100)), # 100,200
    'anyway__max_depth': np.array(range(3,20)), # 6,8,10,12
    'anyway__min_samples_leaf': np.array(range(3,20)), # 3,5,7,10
    'anyway__min_samples_split': np.array(range(3,20)), # 2,3,5,10
    'anyway__n_jobs':[-1]} # -1
]



# 2. 모델
# 스케일러+모델+랜덤서치
# 파이프라인 = 스케일링을 CV에 엮어주는 기능
# pipe = make_pipeline(MinMaxScaler(), SVC())
pipe = Pipeline([('scaler', MinMaxScaler()),('anyway', RandomForestRegressor())])

model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=2)

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가
print('acc:', model.score(x_test, y_test))

print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)

# acc: 0.8814631962419204
# 최적의 매개변수: Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('anyway',
#                  RandomForestRegressor(max_depth=17, min_samples_leaf=3,
#                                        min_samples_split=8, n_jobs=-1))])
# 최적의 파라미터: {'anyway__n_jobs': -1, 
# 'anyway__n_estimators': 100, 
# 'anyway__min_samples_split': 8, 
# 'anyway__min_samples_leaf': 3, 
# 'anyway__max_depth': 17}


# acc: 0.8831547926909838
# 최적의 매개변수: Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('anyway',
#                  RandomForestRegressor(max_depth=10, min_samples_leaf=3,
#                                        min_samples_split=7, n_estimators=200,
#                                        n_jobs=-1))])
# 최적의 파라미터: {'anyway__n_jobs': -1, 
# 'anyway__n_estimators': 200, 
# 'anyway__min_samples_split': 7, 
# 'anyway__min_samples_leaf': 3, 
# 'anyway__max_depth': 10}



# acc: 0.8796902631228478
# 최적의 매개변수: Pipeline(memory=None,
#          steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
#                 ('anyway',
#                  RandomForestRegressor(bootstrap=True, ccp_alpha=0.0,
#                                        criterion='mse', max_depth=42,
#                                        max_features='auto', max_leaf_nodes=None,
#                                        max_samples=None,
#                                        min_impurity_decrease=0.0,
#                                        min_impurity_split=None,
#                                        min_samples_leaf=4, min_samples_split=7,
#                                        min_weight_fraction_leaf=0.0,
#                                        n_estimators=600, n_jobs=-1,
#                                        oob_score=False, random_state=None,
#                                        verbose=0, warm_start=False))],
#          verbose=False)
# 최적의 파라미터: {'anyway__n_jobs': -1, 
# 'anyway__n_estimators': 600, 
# 'anyway__min_samples_split': 7, 
# 'anyway__min_samples_leaf': 4, 
# 'anyway__max_depth': 42}
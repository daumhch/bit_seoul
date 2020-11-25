# 머신러닝
# 준수한 성능, 빠른 속도


# 기존 keras 모델 만드는 순서
# 1.데이터
# 1.1 load_data
# 1.2 train_test_split
# 1.3 scaler
# 1.4 reshape
# 2.모델
# 3.컴파일 훈련
# 4.평가 예측


# 머신러닝 모델 만드는 순서
# 1.데이터
# 1.1 load_data
# 1.2 train_test_split
# 2.모델+Pipeline+SearchCV
# 3.훈련
# 4.평가 예측

# scaler, reshape, compile이 빠진다
# 머신러닝은 2차원만
# scaler는 Pipeline으로 들어감
# compile은 안 해도 됨(곧바로 fit)
# 머신러닝은 One Hot Encoding도 필요 없다
# 머신러닝 모델 함수 이름에,
# 분류 = Classifier / 회귀 = Regressor가 붙는다

# 1.데이터
# 1.1 load_data
import numpy as np
from sklearn.datasets import load_iris
datasets = load_iris()
x = datasets.data
y = datasets.target
print("========== 데이터 로딩 끝 ==========")

# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, test_size=0.2, random_state=44)
print("========== train_test_split 끝 ==========")

# 2.모델+Pipeline+SearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

parameters = [
    {'anyway__n_estimators':np.array(range(100,1000,25)), # 100,200
    'anyway__max_depth': np.array(range(2,25)), # 6,8,10,12
    'anyway__min_samples_leaf': np.array(range(2,25)), # 3,5,7,10
    'anyway__min_samples_split': np.array(range(2,25)), # 2,3,5,10
    'anyway__n_jobs':[-1]} # -1
]
kfold = KFold(n_splits=5, shuffle=True)
pipe = Pipeline([('scaler', StandardScaler()),('anyway', RandomForestClassifier())])
model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=2)

# 3.훈련
model.fit(x_train, y_train)

# 4.평가 예측
print('acc:', model.score(x_test, y_test))

print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)

# acc: 0.9333333333333333
# 최적의 매개변수: Pipeline(steps=[('scaler', StandardScaler()),
#                 ('anyway',
#                  RandomForestClassifier(max_depth=15, min_samples_leaf=21,
#                                         min_samples_split=7, n_estimators=450,
#                                         n_jobs=-1))])
# 최적의 파라미터: {'anyway__n_jobs': -1, 
# 'anyway__n_estimators': 450, 
# 'anyway__min_samples_split': 7, 
# 'anyway__min_samples_leaf': 21, 
# 'anyway__max_depth': 15}

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("accuracy:",accuracy_score(y_test,y_predict))
# accuracy: 0.9333333333333333

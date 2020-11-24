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

wine = pd.read_csv('./data/csv/winequality-white.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=';' # 구분 기호
                        )
y = wine['quality']
x = wine.drop('quality', axis=1).to_numpy()
print(x.shape)
print(y.shape)

# 1.2 train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


parameters = [
    {'anyway__n_estimators':np.array(range(100,1000,25)), # 100,200
    'anyway__max_depth': np.array(range(1,25)), # 6,8,10,12
    'anyway__min_samples_leaf': np.array(range(1,25)), # 3,5,7,10
    'anyway__min_samples_split': np.array(range(1,25)), # 2,3,5,10
    'anyway__n_jobs':[-1]} # -1
]



# 2. 모델
# 스케일러+모델+랜덤서치
# 파이프라인 = 스케일링을 CV에 엮어주는 기능
# pipe = make_pipeline(MinMaxScaler(), SVC())
pipe = Pipeline([('scaler', StandardScaler()),('anyway', RandomForestClassifier())])

model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=2)

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가
print('acc:', model.score(x_test, y_test))

print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)

'''
parameters = [
    {'anyway__n_estimators':np.array(range(100,500,100)), # 100,200
    'anyway__max_depth': np.array(range(1,20)), # 6,8,10,12
    'anyway__min_samples_leaf': np.array(range(1,20)), # 3,5,7,10
    'anyway__min_samples_split': np.array(range(1,20)), # 2,3,5,10
    'anyway__n_jobs':[-1]} # -1
]
acc: 0.676530612244898
최적의 매개변수: Pipeline(steps=[('scaler', MinMaxScaler()),
                ('anyway',
                 RandomForestClassifier(max_depth=17, min_samples_split=12,
                                        n_estimators=300, n_jobs=-1))])
최적의 파라미터: {'anyway__n_jobs': -1, 
'anyway__n_estimators': 300, 
'anyway__min_samples_split': 12, 
'anyway__min_samples_leaf': 1, 
'anyway__max_depth': 17}



parameters = [
    {'anyway__n_estimators':np.array(range(100,1000,100)), # 100,200
    'anyway__max_depth': np.array(range(1,50)), # 6,8,10,12
    'anyway__min_samples_leaf': np.array(range(1,50)), # 3,5,7,10
    'anyway__min_samples_split': np.array(range(1,50)), # 2,3,5,10
    'anyway__n_jobs':[-1]} # -1
]
acc: 0.6438775510204081
최적의 매개변수: Pipeline(steps=[('scaler', StandardScaler()),
                ('anyway',
                 RandomForestClassifier(max_depth=10, min_samples_split=6,
                                        n_estimators=400, n_jobs=-1))])
최적의 파라미터: {'anyway__n_jobs': -1, 
'anyway__n_estimators': 400, 
'anyway__min_samples_split': 6, 
'anyway__min_samples_leaf': 1, 
'anyway__max_depth': 10}




parameters = [
    {'anyway__n_estimators':np.array(range(100,1000,25)), # 100,200
    'anyway__max_depth': np.array(range(1,25)), # 6,8,10,12
    'anyway__min_samples_leaf': np.array(range(1,25)), # 3,5,7,10
    'anyway__min_samples_split': np.array(range(1,25)), # 2,3,5,10
    'anyway__n_jobs':[-1]} # -1
]
acc: 0.6602040816326531
최적의 매개변수: Pipeline(steps=[('scaler', StandardScaler()),
                ('anyway',
                 RandomForestClassifier(max_depth=16, min_samples_leaf=4,
                                        min_samples_split=13, n_estimators=750,
                                        n_jobs=-1))])
최적의 파라미터: {'anyway__n_jobs': -1, 
'anyway__n_estimators': 750, 
'anyway__min_samples_split': 13, 
'anyway__min_samples_leaf': 4, 
'anyway__max_depth': 16}

'''



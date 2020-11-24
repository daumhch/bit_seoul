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

iris = pd.read_csv('./data/csv/iris_ys.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=',' # 구분 기호
                        )
x = iris.iloc[:,0:4]
y = iris.iloc[:,4]

# 1.2 train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

# parameters = [
#     {"C":[1,10,100,1000],"kernel":["linear"]}, # 4x1번
#     {"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[0.001,0.0001]}, # 4x1x2번
#     {"C":[1,10,100,1000],"kernel":["sigmoid"],"gamma":[0.001,0.0001]} # 4x1x2번
# ] # 총 4+8+8=20번

parameters = [
    {'anyway__C':[1,10,100,1000],'anyway__kernel':['linear']}, # 4x1번
    {'anyway__C':[1,10,100,1000],'anyway__kernel':['rbf'],'anyway__gamma':[0.001,0.0001]}, # 4x1x2번
    {'anyway__C':[1,10,100,1000],'anyway__kernel':['sigmoid'],'anyway__gamma':[0.001,0.0001]} # 4x1x2번
] # 총 4+8+8=20번



# 2. 모델
# 스케일러+모델+랜덤서치
# 파이프라인 = 스케일링을 CV에 엮어주는 기능
# pipe = make_pipeline(MinMaxScaler(), SVC())
pipe = Pipeline([('scaler', MinMaxScaler()),('anyway', SVC())])

model = RandomizedSearchCV(pipe, parameters, cv=5)

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가
print('acc:', model.score(x_test, y_test))
# acc: 0.9666666666666667

print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)

# acc: 0.9333333333333333
# 최적의 매개변수: Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('anyway', SVC(C=1, kernel='linear'))])
# 최적의 파라미터: {'anyway__kernel': 'linear', 
# 'anyway__C': 1}





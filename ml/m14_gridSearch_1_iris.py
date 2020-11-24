# 분류

# 클래스파이어 모델들 추출

# 1.데이터
# 1.1 load_data
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


# 2. 모델
kfold = KFold(n_splits=5, shuffle=True)

# model = SVC()
parameters = [
    {"C":[1,10,100,1000],"kernel":["linear"]}, # 4x1번
    {"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[0.001,0.0001]}, # 4x1x2번
    {"C":[1,10,100,1000],"kernel":["sigmoid"],"gamma":[0.001,0.0001]} # 4x1x2번
] # 총 4+8+8=20번
model = GridSearchCV(SVC(), parameters, cv=kfold) # kfold가 5번 x 20번 = 총 100번


# 3. 훈련
model.fit(x_train,y_train)


# 4. 평가 예측
print("최적의 매개변수:", model.best_estimator_)

y_predict = model.predict(x_test)
print('최종정답률:',accuracy_score(y_test, y_predict))



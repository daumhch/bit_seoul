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


# 2. 모델
# 스케일러+모델
# 파이프라인 = 스케일링을 CV에 엮어주는 기능
# pipe = make_pipeline(MinMaxScaler(), SVC())
pipe = Pipeline([("scaler", MinMaxScaler()),('anyway', SVC())])


# 3. 훈련
pipe.fit(x_train, y_train)


# 4. 평가
print('acc:', pipe.score(x_test, y_test))
# acc: 0.9666666666666667


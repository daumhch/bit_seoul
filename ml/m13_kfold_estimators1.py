# 분류

# 클래스파이어 모델들 추출

# 1.데이터
# 1.1 load_data
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings

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
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


allAlgorithms = all_estimators(type_filter='classifier')

from sklearn.model_selection import KFold, cross_val_score

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        kfold = KFold(n_splits=5, shuffle=True)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        y_pred = model.predict(x_test)
        print(kfold.n_splits,"/", name, '의 정답률:', format(accuracy_score(y_test, y_pred), '.6f'))
        print(kfold.n_splits,"/", name, '의 scores:', scores)
    except:
        print(name,'은 수행될 수 없습니다')
        pass
# 원래 50여개의 모델이 나와야 하는데, 버전바뀌고 에러가 발생한다





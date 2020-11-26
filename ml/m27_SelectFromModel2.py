# 실습
# 1. 상단 모델에 그리드서치 또는 랜덤 서치 적용
# 최적의 R2값과 피처임포턴츠 구할 것

# 2. 위 쓰레드 값으로 SelectFromModel을 구해서
# 최적의 피쳐 갯수를 구할 것

# 3. 위 피처 갯수로 데이터(피처)를 수정(삭제)하여
# 그리드서치 또는 랟넘 서치 적용해서
# 최적의 R2값을 구할 것

# 1번값과 2번값을 비교

print("========== 데이터 로딩 시작 ==========")

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score
x, y = load_boston(return_X_y=True)
print("original x.shape:",x.shape)
print("========== 데이터 로딩 끝 ==========")


print("========== train_test_split 시작 ==========")
# train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, test_size=0.2, random_state=44)
print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)
print("========== train_test_split 끝 ==========")


# 2.모델+Pipeline+SearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
parameters = [
    {'anyway__n_estimators':np.array(range(100,1000,100)),
    'anyway__learning_rate':np.array(np.arange(0.01,1,0.1)),
    'anyway__max_depth': np.array(range(3,10)),
    'anyway__colsample_bytree': np.array(np.arange(0.5,1,0.1)),
    'anyway__colsample_bylevel': np.array(np.arange(0.5,1,0.1)),
    'anyway__n_jobs':[-1]
    }
]
kfold = KFold(n_splits=5, shuffle=True)
pipe = Pipeline([('scaler', StandardScaler()),('anyway', XGBRegressor())])
model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=0)

# 3.훈련
model.fit(x_train, y_train)

# 4.평가 예측
score = model.score(x_test, y_test)
print("original R2:", score)
original_params = model.best_params_
print("최적의 파라미터:", original_params)
# original R2: 0.8696630894380475
# 최적의 파라미터: {'anyway__n_jobs': -1, 'anyway__n_estimators': 400, 'anyway__max_depth': 6, 'anyway__learning_rate': 0.31000000000000005, 'anyway__colsample_bytree': 0.8999999999999999, 'anyway__colsample_bylevel': 0.6}




###### 최적의 파라미터로, 다시 모델 돌리기 -> 피쳐임포턴스 구하기 위해서
model = XGBRegressor(parameters=original_params)
# 3.훈련
model.fit(x_train, y_train)

# 4.평가 예측
score = model.score(x_test, y_test)
print("find param R2:", score)
print("find f.i:",model.feature_importances_)
# find param R2: 0.8902902220270855
# find f.i: [0.01311134 0.00178977 0.00865051 0.00337766 0.03526587 0.24189197
#  0.00975884 0.06960727 0.01454236 0.03254252 0.04658296 0.00757505
#  0.51530385]



thresholds = np.sort(model.feature_importances_) # default 오름차순
# print(thresholds)
# print(type(thresholds))

temp_array =[]
for thresh in thresholds:
    selection = SelectFromModel(model,threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print('Thresh=%.6f, n=%d, R2:%.4f' 
            %(thresh, select_x_train.shape[1], score*100.0))
    temp_array.append([thresh, score])

# temp_array를 R2 기준으로 오름차순 정렬하고,
# 마지막 값이 최대 R2일 때의 thresh를 적용
print("temp_array:\r\n", temp_array)
temp_array.sort(key=lambda x: x[1])
print("temp_array:\r\n", temp_array)

feature_thresh = temp_array[-1][0]

def earseLowFI_index(fi_arr, low_value, input_arr):
    input_arr = input_arr.T
    temp = []
    for i in range(fi_arr.shape[0]):
        if fi_arr[i] >= low_value:
            temp.append(input_arr[i,:])
    temp = np.array(temp)
    temp = temp.T
    return temp

print("before x.shape:",x.shape)
x = earseLowFI_index(model.feature_importances_, feature_thresh, x)
print("after x.shape:",x.shape)



print("========== train_test_split 시작 ==========")
# train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, test_size=0.2, random_state=44)
print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)
print("========== train_test_split 끝 ==========")


# 2.모델+Pipeline+SearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
parameters = [
    {'anyway__n_estimators':np.array(range(100,1000,100)),
    'anyway__learning_rate':np.array(np.arange(0.01,1,0.1)),
    'anyway__max_depth': np.array(range(3,10)),
    'anyway__colsample_bytree': np.array(np.arange(0.5,1,0.1)),
    'anyway__colsample_bylevel': np.array(np.arange(0.5,1,0.1)),
    'anyway__n_jobs':[-1]
    }
]
kfold = KFold(n_splits=5, shuffle=True)
pipe = Pipeline([('scaler', StandardScaler()),('anyway', XGBRegressor())])
model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=0)

# 3.훈련
model.fit(x_train, y_train)

# 4.평가 예측
score = model.score(x_test, y_test)
print("after cutting R2:", score)
original_params = model.best_params_
print("after cutting 최적의 파라미터:", original_params)

# after cutting R2: 0.9153494983907459
# after cutting 최적의 파라미터: {'anyway__n_jobs': -1, 'anyway__n_estimators': 400, 'anyway__max_depth': 3, 'anyway__learning_rate': 0.31000000000000005, 'anyway__colsample_bytree': 0.8999999999999999, 'anyway__colsample_bylevel': 0.8999999999999999}



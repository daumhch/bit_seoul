import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score

x, y = load_boston(return_X_y=True)

print("========== train_test_split 시작 ==========")
# train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, test_size=0.2, random_state=44)
print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)
print("========== train_test_split 끝 ==========")


model = XGBRegressor(n_jobs=-1)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("R2:", score)

thresholds = np.sort(model.feature_importances_) # default 오름차순
print(thresholds)
print(type(thresholds))

import time

start1 = time.time()

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    selection_model = XGBRegressor(n_jobs=6)
    selection_model.fit(select_x_train, y_train)
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    # print('Thresh=%.6f, n=%d, R2:%.4f' 
    #         %(thresh, select_x_train.shape[1], score*100.0))

start2 = time.time()

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    # print('Thresh=%.6f, n=%d, R2:%.4f' 
    #         %(thresh, select_x_train.shape[1], score*100.0))

end = start2-start1
print('그냥 걸린 시간:', end)

end2 = time.time()-start2
print('잡스 걸린 시간:', end2)


'''
n_jobs 세팅 별 걸린 시간(초)
-1: 2.889275550842285
1: 1.3045387268066406
6: 1.1529178619384766

-1보다 6이 빠르다

1보다 6이 빠르다

'''



# SelectFromModel을 돌리고
# 가장 좋은 모델만 저장하자



import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score

x, y = load_iris(return_X_y=True)

print("========== train_test_split 시작 ==========")
# train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, test_size=0.2, random_state=44)
print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)
print("========== train_test_split 끝 ==========")


model = XGBClassifier(n_jobs=6)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("R2:", score)

thresholds = np.sort(model.feature_importances_) # default 오름차순
print(thresholds)
print(type(thresholds))

import joblib


best_score = 0.0
for thresh in thresholds:
    selection = SelectFromModel(model,threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    selection_model = XGBClassifier(n_jobs=6)
    selection_model.fit(select_x_train, y_train)
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    if best_score<score:
        best_score = score
        joblib.dump(model, open('./save/xgb_save/m37_2_best_model.joblib.dat','wb'))

    print('Thresh=%.6f, n=%d, R2:%.4f' 
            %(thresh, select_x_train.shape[1], score*100.0))

print('best_score:',best_score)


model3 = joblib.load(open('./save/xgb_save/m37_2_best_model.joblib.dat','rb'))
y_predict3 = model3.predict(x_test)
score3 = accuracy_score(y_test, y_predict3)
print("score3:", score3)


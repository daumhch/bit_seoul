# 다중분류

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

# 회귀 지표
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_log_error, r2_score

# 분류 지표
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import log_loss, roc_auc_score

x, y = load_breast_cancer(return_X_y=True)

x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


# 모델
# model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model = XGBClassifier(learning_rate=0.1)

# 훈련
model.fit(x_train, y_train, 
            verbose=True, # fit 과정 표시
            eval_metric=['logloss', 'error', 'auc'], # 기준 지표
            eval_set=[(x_test,y_test)]) # 테스트 데이터셋

# eval_metric 평가지표들
# rmse, mae, logloss, error, auc 

# 평가 예측
results = model.evals_result()
# 결과 type이 dictionary, 마지막 값은 아래와 같이 접근할 수 있다
print("eval's result logloss:", results['validation_0']['logloss'][-1])
print("eval's result error:", results['validation_0']['error'][-1])
print("eval's result auc:", results['validation_0']['auc'][-1])

y_pred = model.predict(x_test)


accuracy = accuracy_score(y_pred,y_test)
print("accuracy:",accuracy)

precision_score = precision_score(y_pred,y_test)
print("precision_score:",precision_score)

recall_score = recall_score(y_pred,y_test)
print("recall_score:",recall_score)

f1_score = f1_score(y_pred,y_test)
print("f1_score:",f1_score)

log_loss_score = log_loss(y_pred,y_test)
print("log_loss_score:",log_loss_score)

roc_auc_score_score = roc_auc_score(y_pred,y_test)
print("roc_auc_score_score:",roc_auc_score_score)



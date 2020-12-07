
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score


x, y = load_boston(return_X_y=True)

x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


# 모델
model = XGBRegressor(n_estimators=10, learning_rate=0.1)


# 훈련
model.fit(x_train, y_train, verbose=True, 
            eval_metric=['rmse'], # 기준 지표
            eval_set=[(x_test,y_test)]) # 테스트 데이터셋








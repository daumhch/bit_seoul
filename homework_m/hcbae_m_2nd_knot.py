# KFold, SearchCV
# feature_importances_
# PCA
# 를 모두 적용한 소스 만들기


# 1.데이터
# 1.1 load_data
print("========== 데이터 로딩 시작 ==========")
import numpy as np
from sklearn.datasets import load_diabetes
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print("original x.shape:",x.shape)
print("========== 데이터 로딩 끝 ==========")


print("========== feature importance cutting 시작 ==========")
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)
model = XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances(data_name, model):
    n_features = data_name.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align='center')
    plt.yticks(np.arange(n_features), data_name.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
print("fi median:",np.median(model.feature_importances_))
print("fi mean:",np.mean(model.feature_importances_))
plt.axvline(x=np.median(model.feature_importances_), color='r', linestyle='-', linewidth=2)
plt.axvline(x=np.mean(model.feature_importances_), color='b', linestyle='-', linewidth=2)
plot_feature_importances(datasets, model)
plt.show()
import numpy as np
def earseLowFI_index(fi_arr, low_value, input_arr):
    input_arr = input_arr.T
    temp = []
    for i in range(fi_arr.shape[0]):
        if fi_arr[i] >= low_value:
            temp.append(input_arr[i,:])
    temp = np.array(temp)
    temp = temp.T
    return temp

print("before f.i.cutting x.shape:",x.shape)
print("f.i:",model.feature_importances_)
x = earseLowFI_index(model.feature_importances_, 0.085, x)
print("after f.i.cutting x.shape:",x.shape)
print("========== feature importance cutting 끝 ==========")


print("========== PCA를 위한 Scaler + PCA 시작 ==========")
# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)


# 1.5 PCA
from sklearn.decomposition import PCA
cumsum_standard = 0.95
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= cumsum_standard) +1
pca = PCA(n_components=d)
x = pca.fit_transform(x)
print("after pca x.shape", x.shape) # (70000, 202)
print("========== PCA를 위한 Scaler + PCA 끝 ==========")


print("========== train_test_split 시작 ==========")
# 1.2 train_test_split
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
print('acc:', model.score(x_test, y_test))

print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)


from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
print('최종정답률:',r2_score(y_test, y_predict))
# accuracy: 0.9333333333333333


# acc: 0.460131913632567
# 최적의 매개변수: Pipeline(steps=[('scaler', StandardScaler()),
#                 ('anyway',
#                  XGBRegressor(base_score=0.5, booster='gbtree',
#                               colsample_bylevel=0.7999999999999999,      
#                               colsample_bynode=1,
#                               colsample_bytree=0.8999999999999999, gamma=0,
#                               gpu_id=-1, importance_type='gain',
#                               interaction_constraints='', learning_rate=0.01,
#                               max_delta_step=0, max_depth=4, min_child_weight=1,
#                               missing=nan, monotone_constraints='()',    
#                               n_estimators=700, n_jobs=-1, num_parallel_tree=1,
#                               random_state=0, reg_alpha=0, reg_lambda=1, 
#                               scale_pos_weight=1, subsample=1,
#                               tree_method='exact', validate_parameters=1,
#                               verbosity=None))])
# 최적의 파라미터: {'anyway__n_jobs': -1, 'anyway__n_estimators': 700, 'anyway__max_depth': 4, 'anyway__learning_rate': 0.01, 'anyway__colsample_bytree': 0.8999999999999999, 'anyway__colsample_bylevel': 0.7999999999999999}
# 최종정답률: 0.460131913632567





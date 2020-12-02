import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

import numpy as np
import pandas as pd
import timeit

#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

start_time = timeit.default_timer() # 시작 시간 체크

# ======== 데이터 불러오기 시작 ========
indexes = np.load('./project2/csv_index.npy', allow_pickle=True)
x = np.load('./project2/merge_data.npy', allow_pickle=True)
y = np.load('./project2/merge_target.npy', allow_pickle=True)
# print(x[0])
print("npy x.shape:",x.shape)
print("npy y.shape:",y.shape)

# 테스트를 위해 잘라냄
# x = x[:10000,:]
# y = y[:10000]
# 그냥 자르기 보다는 솎아내는게 낫겠다
# temp_x,x, temp_y,y = train_test_split(
#     x,y, random_state=44, shuffle=True, test_size=0.5)

# print("merge_index:", indexes)
print("data use.shape:",x.shape)
print("target use.shape:",y.shape)
# ======== 데이터 불러오기 끝 ========


# ======== 피쳐 임포턴스 특성 찾기 시작 ========
x_train,x_pred, y_train,y_stand = train_test_split(
    x, y, random_state=44, train_size=0.8, test_size=0.2)
# print("split shape x_train/x_test:",x_train.shape, x_test.shape)
# print("split shape y_train/y_test:",y_train.shape, y_test.shape)

x_train,x_test, y_train,y_test = train_test_split(
    x_train, y_train, random_state=44, train_size=0.75, test_size=0.25)
print("split x_train.shape:",x_train.shape)
print("split x_test.shape:",x_test.shape)
print("split x_pred.shape:",x_pred.shape)

parameters_arr = [
    {'anyway__n_jobs': [-1],
    'anyway__n_estimators': np.array(range(100,500,100)),
    'anyway__num_leaves': [10],
    'anyway__max_depth': [5,6,7,8] # 트리 최대 깊이 default 6
    }
]

start_time2 = timeit.default_timer() # 시작 시간 체크

kfold = KFold(n_splits=5, shuffle=True)
pipe = Pipeline([('scaler', StandardScaler()),('anyway', LGBMClassifier() )])
model = RandomizedSearchCV(pipe, parameters_arr, cv=kfold, verbose=0)
model.fit(x_train, y_train)
score_at_fi = model.score(x_test, y_test)
print("original score:", score_at_fi)
original_params_at_fi = model.best_params_
print("original 최적의 파라미터:", original_params_at_fi)

end_time2 = timeit.default_timer() # 시작 시간 체크


###### 최적의 파라미터로, 다시 모델 돌리기 -> 피쳐임포턴스 구하기 위해서
# model = XGBClassifier(original_params_at_fi)
# model.fit(x_train, y_train)
# score_at_fi_param = model.score(x_test, y_test)
# print("find param R2:", score_at_fi_param)
# print("find f.i:",model.feature_importances_)
best_model = model.best_estimator_
score_at_fi_param = best_model.score(x_test, y_test)
print("find param score:", score_at_fi_param)

g_best_model = model.best_estimator_.named_steps["anyway"]
g_feature_importance = model.best_estimator_.named_steps["anyway"].feature_importances_
print("g_feature_importance:\r\n",g_feature_importance)

######## 최적 파라미터로 구한 f.i를 정렬 후, 최대 R2에서 threshold 구하기 
thresholds = np.sort(g_feature_importance) # default 오름차순
temp_array =[]
for thresh in thresholds:
    selection = SelectFromModel(g_best_model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    selection_model = g_best_model
    selection_model.fit(select_x_train, y_train)
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    # print('Thresh=%.6f, n=%d, R2:%.6f' 
    #         %(thresh, select_x_train.shape[1], score))
    temp_array.append([thresh, score])

# temp_array를 R2 기준으로 오름차순 정렬하고,
# 마지막 값이 최대 R2일 때의 thresh를 적용
# print("temp_array:\r\n", temp_array)
temp_array.sort(key=lambda x: x[1])
# print("temp_array:\r\n", temp_array)

feature_thresh = temp_array[-1][0]
print("feature_thresh:",feature_thresh)
######## 최적 파라미터로 구한 f.i를 정렬 후, 최대 R2에서 threshold 구하기 



############ 위에서 구한 최대 R2 기준 Threshold와 F.I 그래프 그리기
import matplotlib.pyplot as plt
import numpy as np
def drawPlt(index, feature_importances, feature_thresh):
    n_features = len(feature_importances)
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.bar(np.arange(n_features), feature_importances)
    plt.ylabel("Feature Importances(log)")
    plt.xlabel("Features")
    plt.xticks(np.arange(n_features), index, rotation=90)
    plt.xlim(-1, n_features)
    plt.yscale('log')
    plt.axhline(y=feature_thresh, color='r')
    # plt.show()
    plt.savefig('./project2/feature_importances.png', bbox_inches='tight', pad_inches=0)
    plt.close()

drawPlt(indexes, g_feature_importance, feature_thresh)
############ 위에서 구한 최대 R2 기준 Threshold와 F.I 그래프 그리기





def earseLowFI_index(fi_arr, low_value, input_arr):
    input_arr = input_arr.T
    temp = []
    for i in range(fi_arr.shape[0]):
        if fi_arr[i] > low_value:
            temp.append(input_arr[i,:])
    temp = np.array(temp)
    temp = temp.T
    return temp


print("before erase low f.i x.shape:",x.shape)
x = earseLowFI_index(g_feature_importance, 0, x)
print("after erase low f.i x.shape:",x.shape)




# ======== PCA 적용 시작 ========
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# 1.5 PCA
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum:\r\n", cumsum)

plt.rcParams["figure.figsize"] = (20, 10)
plt.plot(cumsum, marker='.')
plt.xlim([25,x.shape[1]])
plt.ylim([0.95,1])
plt.grid()
plt.savefig('./project2/pca_cumsum.png', bbox_inches='tight', pad_inches=0)
plt.close()

cumsum_standard = 0.99
d = np.argmax(cumsum >= cumsum_standard) +1
print("n_components:",d)
pca = PCA(n_components=d)
x = pca.fit_transform(x)
print("after pca x.shape", x.shape)
# ======== PCA 적용 끝 ========




# 모델 돌리자
print("x.shape", x.shape)
print("y.shape", y.shape)



# ======== 모델을 위한 train_test_split 시작 ========
x_train,x_pred, y_train,y_stand = train_test_split(
    x, y, random_state=44, train_size=0.8, test_size=0.2)
# print("split shape x_train/x_test:",x_train.shape, x_test.shape)
# print("split shape y_train/y_test:",y_train.shape, y_test.shape)

x_train,x_test, y_train,y_test = train_test_split(
    x_train, y_train, random_state=44, train_size=0.75, test_size=0.25)
print("split x_train.shape:",x_train.shape)
print("split x_test.shape:",x_test.shape)
print("split x_pred.shape:",x_pred.shape)
# ======== 모델을 위한 train_test_split 끝 ========


start_time3 = timeit.default_timer() # 시작 시간 체크

# ======== 최적 파라미터 적용 모델+Pipeline+SearchCV 시작 ========
pipe = Pipeline([('scaler', StandardScaler()),('anyway', LGBMClassifier() )])
model = RandomizedSearchCV(pipe, parameters_arr, cv=kfold, verbose=0)
model.fit(x_train, y_train)
score_at_final = model.score(x_test, y_test)
print("final score:", score_at_final)

score_at_final_param = model.best_estimator_.score(x_test, y_test)
print("final param score:", score_at_final_param)
# ======== 최적 파라미터 적용 모델+Pipeline+SearchCV 끝 ========

end_time3 = timeit.default_timer() # 시작 시간 체크



y_predict = model.best_estimator_.predict(x_pred)
print("================================")
print("original score:", score_at_fi)
print("find param score:", score_at_fi_param)
print("final score:", score_at_final)
print("final param score:", score_at_final_param)
print('따로 빼낸 pred로 만든 accuracy:',accuracy_score(y_stand, y_predict))
print("================================")

print("1st 탐색 %f초 걸렸습니다." % (end_time2 - start_time2)) 
print("2nd 탐색 %f초 걸렸습니다." % (end_time3 - start_time3))


terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time)) 




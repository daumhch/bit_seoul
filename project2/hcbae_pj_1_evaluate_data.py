import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

import numpy as np
import pandas as pd
import timeit

from xgboost import XGBClassifier

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
x = np.load('./project2/iv_data.npy', allow_pickle=True)
y = np.load('./project2/iv_target.npy', allow_pickle=True)
# print(x[0])
print("npy x.shape:",x.shape)
print("npy y.shape:",y.shape)

# 테스트를 위해 잘라냄
# x = x[:10000,:]
# y = y[:10000]
# 그냥 자르기 보다는 솎아내는게 낫겠다
temp_x,x, temp_y,y = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.5)

# print("merge_index:", indexes)
print("merge_data.shape:",x.shape)
print("merge_target.shape:",y.shape)
# ======== 데이터 불러오기 끝 ========


# ======== 피쳐 임포턴스 특성 찾기 시작 ========
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

parameters_arr = [
    {'anyway__n_jobs':[-1],
    'anyway__nthread':[-1],
    'anyway__n_estimators':np.array(range(100,1000,100)),
    'anyway__max_depth': np.array(range(3,10)), # 트리 최대 깊이 default 6
    'anyway__learning_rate':np.array(np.arange(0.1,0.6,0.1)), # 학습률 default 0.3
    'anyway__colsample_bytree': np.array(np.arange(0.7,1,0.1)), # 트리생성시 반영비율 default 1
    'anyway__colsample_bylevel': np.array(np.arange(0.7,1,0.1)) # 레벨생성시 반영비율 default 1
    }
]
kfold = KFold(n_splits=5, shuffle=True)
xbg = XGBClassifier(objective='multi:softmax', eval_metric='logloss')
pipe = Pipeline([('scaler', StandardScaler()),('anyway', xbg)])
model = RandomizedSearchCV(pipe, parameters_arr, cv=kfold, verbose=0)
model.fit(x_train, y_train)
score_at_fi = model.score(x_test, y_test)
print("original R2:", score_at_fi)
original_params_at_fi = model.best_params_
print("original 최적의 파라미터:", original_params_at_fi)


###### 최적의 파라미터로, 다시 모델 돌리기 -> 피쳐임포턴스 구하기 위해서
model = XGBClassifier(original_params_at_fi)
model.fit(x_train, y_train)
score_at_fi_param = model.score(x_test, y_test)
print("find param R2:", score_at_fi_param)
print("find f.i:",model.feature_importances_)


######## 최적 파라미터로 구한 f.i를 정렬 후, 최대 R2에서 threshold 구하기 
thresholds = np.sort(model.feature_importances_) # default 오름차순
temp_array =[]
for thresh in thresholds:
    selection = SelectFromModel(model,threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    selection_model = XGBClassifier(original_params_at_fi)
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
    n_features = len(model.feature_importances_)
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

drawPlt(indexes, model.feature_importances_, feature_thresh)
############ 위에서 구한 최대 R2 기준 Threshold와 F.I 그래프 그리기





def earseLowFI_index(fi_arr, low_value, input_arr):
    input_arr = input_arr.T
    temp = []
    for i in range(fi_arr.shape[0]):
        if fi_arr[i] >= low_value:
            temp.append(input_arr[i,:])
    temp = np.array(temp)
    temp = temp.T
    return temp


print("before erase low f.i x.shape:",x.shape)
x = earseLowFI_index(model.feature_importances_, 0.001, x)
print("after erase low f.i x.shape:",x.shape)




# ======== PCA 적용 시작 ========
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)



# 1.5 PCA
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print("cumsum:\r\n", cumsum)

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

import pickle as pk
pk.dump(pca, open("./project2/pca.pkl","wb"))

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


###### 최적의 파라미터로, 다시 모델 돌리기 -> 모델 평가를 위해서
model = XGBClassifier(original_params_at_fi)
model.fit(x_train, y_train)
score_at_final_param = model.score(x_test, y_test)
print("final param R2:", score_at_final_param)


y_predict = model.predict(x_pred)
print("================================")
print("original score:", score_at_fi)
print("find param score:", score_at_fi_param)
print("final param score:", score)
print('따로 빼낸 pred로 만든 accuracy:',accuracy_score(y_stand, y_predict))
print("================================")


###### 추가 평가
from sklearn.metrics import classification_report
print("classification_report:\r\n",classification_report(y_test, y_predict))






terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time)) 







import numpy as np
import timeit

start_time = timeit.default_timer() # 시작 시간 체크

# ======== 데이터 불러오기 시작 ========
indexes = np.load('./project/merge_index.npy', allow_pickle=True)
x = np.load('./project/merge_data.npy', allow_pickle=True)
y = np.load('./project/merge_target.npy', allow_pickle=True)
print("npy x.shape:",x.shape)
print("npy y.shape:",y.shape)

# 테스트를 위해 잘라냄
# x = x[:10000,:]
# y = y[:10000]
# 그냥 자르기 보다는 솎아내는게 낫겠다
from sklearn.model_selection import train_test_split
temp_x,x, temp_y,y = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.01)

print("merge_index:", indexes)
print("merge_data.shape:",x.shape)
print("merge_target.shape:",y.shape)
# ======== 데이터 불러오기 끝 ========



# ======== 피쳐 임포턴스 특성 찾기 시작 ========
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

parameters_arr = [
    {'anyway__n_estimators':np.array(range(100,1000,100)),
    'anyway__learning_rate':np.array(np.arange(0.1,0.6,0.1)), # 학습률 default 0.3
    'anyway__max_depth': np.array(range(3,10)), # 트리 최대 깊이 default 6
    'anyway__colsample_bytree': np.array(np.arange(0.7,1,0.1)), # 트리생성시 반영비율 default 1
    'anyway__colsample_bylevel': np.array(np.arange(0.7,1,0.1)), # 레벨생성시 반영비율 default 1
    'anyway__n_jobs':[-1]
    }
]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
kfold = KFold(n_splits=5, shuffle=True)
xbg = XGBClassifier(tree_method = "auto", eval_metric='rmse')
pipe = Pipeline([('scaler', StandardScaler()),('anyway', xbg)])
model = RandomizedSearchCV(pipe, parameters_arr, cv=kfold, verbose=0, random_state=44)
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


from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
thresholds = np.sort(model.feature_importances_) # default 오름차순
temp_array =[]
for thresh in thresholds:
    selection = SelectFromModel(model,threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    selection_model = XGBClassifier(n_jobs=-1)
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
x = earseLowFI_index(model.feature_importances_, feature_thresh, x)
print("after erase low f.i x.shape:",x.shape)
# ======== 피쳐 임포턴스 특성 찾기 끝 ========



# ======== PCA 적용 시작 ========
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# 1.5 PCA
from sklearn.decomposition import PCA
cumsum_standard = 0.95
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print("cumsum:\r\n", cumsum)
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
from sklearn.model_selection import train_test_split 
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



# ======== y데이터 확인 후 원핫인코딩 시작 ========
# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# ======== y데이터 확인 후 원핫인코딩 끝 ========




# ======== 모델+Pipeline+SearchCV 시작 ========
parameters_arr2 = [
    {'anyway__n_estimators':np.array(range(100,1000,100)),
    'anyway__learning_rate':np.array(np.arange(0.1,0.6,0.1)), # 학습률 default 0.3
    'anyway__max_depth': np.array(range(3,10)), # 트리 최대 깊이 default 6
    'anyway__colsample_bytree': np.array(np.arange(0.7,1,0.1)), # 트리생성시 반영비율 default 1
    'anyway__colsample_bylevel': np.array(np.arange(0.7,1,0.1)), # 레벨생성시 반영비율 default 1
    'anyway__n_jobs':[-1]
    }
]
kfold = KFold(n_splits=5, shuffle=True)
xbg = XGBClassifier(tree_method = "auto", eval_metric='rmse')
pipe = Pipeline([('scaler', StandardScaler()),('anyway', xbg)])
model = RandomizedSearchCV(pipe, parameters_arr2, cv=kfold, verbose=0, random_state=44)
model.fit(x_train, y_train)
score_at_ac = model.score(x_test, y_test)
print("after cutting R2:", score_at_ac)
original_params2 = model.best_params_
print("after cutting 최적의 파라미터:", original_params2)
# ======== 모델+Pipeline+SearchCV 끝 ========



# ======== 최적 파라미터 적용 모델+Pipeline+SearchCV 시작 ========
model = XGBClassifier(original_params2)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("final param R2:", score)
# ======== 최적 파라미터 적용 모델+Pipeline+SearchCV 끝 ========


y_predict = model.predict(x_pred)
print("================================")
print("original score:", score_at_fi)
print("find param score:", score_at_fi_param)
print("after cutting score:", score_at_ac)
print("final param score:", score)
print('따로 빼낸 pred로 만든 accuracy:',accuracy_score(y_stand, y_predict))
print("================================")

print("앞에서 20개만 견본으로 뽑아서 보자")
# print("x_pred[:10]:", x_pred[:20])
print("y_stand[:10]:", y_stand[:20])
print("y_predict[:10]:", y_predict[:20])
print("set(y_test):", set(y_stand))
print("set(y_predict):", set(y_predict))




terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time)) 










# 기준 xgboost
# x_train + x_test 하고 feture importance
# 실행 3번
# 디폴트
# 0인 것 제거 또는 하위 30% 제거
# 3개의 성능 비교

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x = iris.data
y = iris.target
print("init x.shape:",x.shape)

x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

model = XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("acc:", acc)
print(model.feature_importances_)

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

plot_feature_importances(iris, model)
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

print("before x.shape:",x.shape)
x = earseLowFI_index(model.feature_importances_, 0.1, x)
print("after x.shape:",x.shape)



# 여명씨 도움받은 소스
# def get_drop_list(n):
#     a = model.feature_importances_.tolist()
#     b = a.copy()
#     b.sort()
#     b = b[:n] #앞에 몇개
#     index_list = []
#     for i in model.feature_importances_:
#         if i in b:
#             index_list.append(a.index(i))
#     return(index_list)

# drop_list = get_drop_list(3)
# print(drop_list)



x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

model = XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("after erase low fi acc:", acc)


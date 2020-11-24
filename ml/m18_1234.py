from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


model1 = DecisionTreeClassifier(max_depth=4)
model1.fit(x_train, y_train)

model2 = RandomForestClassifier(max_depth=4)
model2.fit(x_train, y_train)

model3 = GradientBoostingClassifier(max_depth=4)
model3.fit(x_train, y_train)

model4 = XGBClassifier(max_depth=4)
model4.fit(x_train, y_train)


# 시각화
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances(data_name, model, model_name):
    n_features = data_name.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align='center')

    plt.yticks(np.arange(n_features), data_name.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel(model_name)
    plt.ylim(-1, n_features)


plt.figure(figsize=(12,9)) # 단위는 찾아보자

plt.subplot(2,2,1) # 2장 중에 첫 번째
plot_feature_importances(cancer, model1, "decisionTree")

plt.subplot(2,2,2) # 2장 중에 두 번째
plot_feature_importances(cancer, model2, "RainForest")

plt.subplot(2,2,3) # 2장 중에 첫 번째
plot_feature_importances(cancer, model3, "GradientBoosting")

plt.subplot(2,2,4) # 2장 중에 두 번째
plot_feature_importances(cancer, model4, "XGB")

plt.show()
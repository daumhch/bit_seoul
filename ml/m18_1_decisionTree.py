from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
print(x.shape)
print(set(y))

x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


model = DecisionTreeClassifier(max_depth=4)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("acc:", acc)
print(model.feature_importances_)

# DecisionTreeClassifier
# acc: 0.9385964912280702
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.00677572 0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.01008994 0.05612587 0.78000877 0.00677572
#  0.00995429 0.         0.         0.13026968 0.         0.        ]



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

plot_feature_importances(cancer, model)
plt.show()




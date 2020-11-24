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
print(x.shape)
print(set(y))

x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(max_depth=4)


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

# RandomForestClassifier
# acc: 0.9736842105263158
# [0.0282948  0.02102581 0.01843328 0.04510397 0.00792051 0.01031183
#  0.0690098  0.15017291 0.00229942 0.00317162 0.00498296 0.00265829
#  0.00817124 0.03191372 0.00145616 0.00236738 0.00248685 0.00148325
#  0.00090421 0.00443486 0.12384482 0.01671214 0.11427714 0.14297743
#  0.01336041 0.00790808 0.03958549 0.11174899 0.00823243 0.00475024]

# GradientBoostingClassifier
# acc: 0.9649122807017544
# [1.24845207e-04 2.19899678e-03 6.02816834e-04 1.48986419e-04
#  5.55291715e-03 2.70382971e-03 1.78242285e-03 2.40448176e-01
#  3.28403529e-03 3.43572016e-04 1.37102038e-03 2.75356895e-03
#  5.85913059e-03 5.75629038e-03 5.97463480e-04 2.13819826e-03
#  1.21928277e-04 2.86804040e-03 9.68148845e-04 1.59375821e-03
#  3.56171580e-02 7.12935085e-02 4.81438691e-01 3.72487614e-02
#  1.39719950e-02 1.58565289e-03 1.07987454e-02 6.59885108e-02
#  6.77979253e-04 1.60851508e-04]

# XGBClassifier
# acc: 0.9649122807017544
# [0.01401363 0.0164375  0.         0.         0.01158929 0.00143937
#  0.0165314  0.17114884 0.00087087 0.00899467 0.00446325 0.00904075
#  0.00537226 0.00438434 0.00347659 0.0042062  0.02468819 0.00048715
#  0.00095482 0.00436991 0.13704772 0.02353887 0.38103914 0.01689425
#  0.00733311 0.00318179 0.02245905 0.09863705 0.00130741 0.00609256]





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





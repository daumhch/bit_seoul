import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target



# # ======== y데이터 확인 후 원핫인코딩 시작 ========
# # OneHotEncoding
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# # ======== y데이터 확인 후 원핫인코딩 끝 ========

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)



print("X_train.shape",X_train.shape)
print("X_test.shape",X_test.shape)
print("y_train.shape",y_train.shape)
print("y_test.shape",y_test.shape)

# Learn to predict each class against the other
# classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)
model = XGBClassifier()
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
print("score:",score)


y_score = model.predict_proba(X_test)
print("y_score.shape:",y_score.shape)


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

model = XGBClassifier(max_depth=4)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("acc:", acc)


# import pickle
# pickle.dump(model, open('./save/xgb_save/cancer.pickle.dat','wb'))
# print("저장 완료")


# model2 = pickle.load(open('./save/xgb_save/cancer.pickle.dat','rb'))

# acc2 = model2.score(x_test, y_test)
# print("acc2:", acc2)

# acc: 0.9649122807017544
# 저장 완료
# acc2: 0.9649122807017544
# 가중치까지 저장된다



# import joblib
# joblib.dump(model, open('./save/xgb_save/cancer.joblib.dat','wb'))
# print("저장 완료")

# model3 = joblib.load(open('./save/xgb_save/cancer.joblib.dat','rb'))
# acc3 = model3.score(x_test, y_test)
# print("acc3:", acc3)

# acc: 0.9649122807017544
# 저장 완료
# acc3: 0.9649122807017544
# joblib도 가중치까지 저장된다



model.save_model('./save/xgb_save/cancer.xgb.model')
print("저장 완료")

model4 = XGBClassifier()
model4.load_model('./save/xgb_save/cancer.xgb.model')
acc4 = model4.score(x_test, y_test)
print("acc4:", acc4)

# acc: 0.9649122807017544
# 저장 완료
# acc4: 0.9649122807017544
# xgb의 경우, 새로 모델을 만들고 불러와야 한다



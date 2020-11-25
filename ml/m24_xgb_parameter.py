# 과적합 방지
# 1. 훈련데이터량을 늘린다
# 2. 피쳐수를 줄인다
# 3. reguraization


from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target
print("x.shape:", x.shape)


from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)



from xgboost import XGBClassifier, XGBRFRegressor, plot_importance
n_estimators = 300 # 생성할 tree의 개수
learning_rate = 1 # 학습 속도
colsample_bytree = 1 # 각 트리마다의 feature 샘플링 비율, 보통 0.5 ~ 1 사용됨
colsample_bylevel = 1 # 각 노드에서 사용되는 기능 (무작위로 선택됨)의 비율
max_depth = 5 # 트리의 최대 깊이, 보통 3-10 사이
n_jobs = -1


model = XGBRFRegressor(max_depth=max_depth, learning_rate=learning_rate,
                        n_estimators=n_estimators, n_jobs=n_jobs,
                        colsample_bylevel = colsample_bylevel,
                        colsample_bytree=colsample_bytree )
# model = XGBRFRegressor()

# score 디폴트로 했던 놈과 성능 비교



model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("acc:", acc)
print(model.feature_importances_)



# n_estimators = 300
# learning_rate = 1
# colsaple_bytree = 1
# colsample_bylevel = 1
# max_depth = 5
# n_jobs = -1
# acc: 0.8656679923522514
# [0.03896952 0.01041784 0.03785317 0.01431077 0.04818305 0.32035738
#  0.01680889 0.10777394 0.01223349 0.02160913 0.04250463 0.01269124
#  0.31628704]



# default XGBRFRegressor()
# acc: 0.8750557059813722
# [0.03221484 0.00531022 0.03028307 0.01112662 0.04654762 0.316874
#  0.0137734  0.10110185 0.01583973 0.02454378 0.04776645 0.01129459
#  0.34332392]



import matplotlib.pyplot as plt
plot_importance(model)
plt.show()



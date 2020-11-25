# 과적합 방지
# 1. 훈련데이터량을 늘린다
# 2. 피쳐수를 줄인다
# 3. reguraization

parameters = [
    {"n_estimators":[100,200,300],"learning_rate":[0.1,0.3,0.001,0.01],
    "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110],"learning_rate":[0.1,0.001,0.01],
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[100,200,300],"learning_rate":[0.1,0.09,1],
    "colsample_bylevel":[0.6,0.7,0.8]}
]


from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target
print("x.shape:", x.shape)


from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)



from xgboost import XGBClassifier, XGBRFRegressor, plot_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score

kfold = KFold(n_splits=5, shuffle=True)


# model = XGBRFRegressor(max_depth=max_depth, learning_rate=learning_rate,
#                         n_estimators=n_estimators, n_jobs=n_jobs,
#                         colsample_bylevel = colsample_bylevel,
#                         colsample_bytree=colsample_bytree )
model = RandomizedSearchCV(XGBRFRegressor(), 
                    parameters, 
                    cv=kfold, 
                    verbose=2) # kfold가 5번 x 20번 = 총 100번

# score 디폴트로 했던 놈과 성능 비교



model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("acc:", acc)





print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)

y_predict = model.predict(x_test)
print('최종정답률:',r2_score(y_test, y_predict))



# RandomizedSearchCV
# acc: 0.8796532540143678
# 최적의 매개변수: XGBRFRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.8,
#                colsample_bytree=1, gamma=0, gpu_id=-1, importance_type='gain',
#                interaction_constraints='', max_delta_step=0, max_depth=6,     
#                min_child_weight=1, missing=nan, monotone_constraints='()',    
#                n_estimators=300, n_jobs=0, num_parallel_tree=300,
#                objective='reg:squarederror', random_state=0, reg_alpha=0,     
#                scale_pos_weight=1, tree_method='exact', validate_parameters=1,
#                verbosity=None)
# 최적의 파라미터: {'n_estimators': 300, 
# 'learning_rate': 1, 'colsample_bylevel': 0.8}
# 최종정답률: 0.8796532540143678




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



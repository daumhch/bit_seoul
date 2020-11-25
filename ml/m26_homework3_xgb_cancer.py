# 과적합 방지
# 1. 훈련데이터량을 늘린다
# 2. 피쳐수를 줄인다
# 3. reguraization


parameters = [
    {"anyway__n_estimators":[100,200,300],"anyway__learning_rate":[0.1,0.3,0.001,0.01],
    "anyway__max_depth":[4,5,6]},
    {"anyway__n_estimators":[90,100,110],"anyway__learning_rate":[0.1,0.001,0.01],
    "anyway__max_depth":[4,5,6], "anyway__colsample_bytree":[0.6,0.9,1]},
    {"anyway__n_estimators":[100,200,300],"anyway__learning_rate":[0.1,0.09,1],
    "anyway__colsample_bylevel":[0.6,0.7,0.8]}
]


from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print("x.shape:", x.shape)


from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)



from xgboost import XGBClassifier, XGBRFRegressor, XGBClassifier, plot_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

kfold = KFold(n_splits=5, shuffle=True)

pipe = Pipeline([('scaler', StandardScaler()),('anyway', XGBClassifier())])

# model = XGBRFRegressor(max_depth=max_depth, learning_rate=learning_rate,
#                         n_estimators=n_estimators, n_jobs=n_jobs,
#                         colsample_bylevel = colsample_bylevel,
#                         colsample_bytree=colsample_bytree )
# model = RandomizedSearchCV(XGBRFRegressor(), 
#                     parameters, 
#                     cv=kfold, 
#                     verbose=2) # kfold가 5번 x 20번 = 총 100번

model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=2)



model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("acc:", acc)
print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)

y_predict = model.predict(x_test)
print('최종정답률:',accuracy_score(y_test, y_predict))



# acc: 0.9736842105263158
# 최적의 매개변수: Pipeline(steps=[('scaler', StandardScaler()),
#                 ('anyway',
#                  XGBClassifier(base_score=0.5, booster='gbtree',        
#                                colsample_bylevel=1, colsample_bynode=1, 
#                                colsample_bytree=1, gamma=0, gpu_id=-1,  
#                                importance_type='gain',
#                                interaction_constraints='', learning_rate=0.1,
#                                max_delta_step=0, max_depth=6,
#                                min_child_weight=1, missing=nan,
#                                monotone_constraints='()', n_estimators=90,
#                                n_jobs=0, num_parallel_tree=1, random_state=0,
#                                reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#                                subsample=1, tree_method='exact',        
#                                validate_parameters=1, verbosity=None))])
# 최적의 파라미터: {'anyway__n_estimators': 90, 'anyway__max_depth': 6, 'anyway__learning_rate': 0.1, 'anyway__colsample_bytree': 1}
# 최종정답률: 0.9736842105263158

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
# print("iris:",iris)
# print(type(iris)) # <class 'sklearn.utils.Bunch'>

x_data = iris.data
y_data = iris.target
# print(type(x_data)) # <class 'numpy.ndarray'>
# print(type(y_data))


np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_y.npy', arr=y_data)





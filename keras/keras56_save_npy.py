import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np

# 나머지 7개를 저장하시오
from tensorflow.keras.datasets import cifar10
(cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()
np.save('./data/cifar10_x_train.npy', arr=cifar10_x_train)
np.save('./data/cifar10_x_test.npy', arr=cifar10_x_test)
np.save('./data/cifar10_y_train.npy', arr=cifar10_y_train)
np.save('./data/cifar10_y_test.npy', arr=cifar10_y_test)


from tensorflow.keras.datasets import fashion_mnist
(fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = fashion_mnist.load_data()
np.save('./data/fashion_x_train.npy', arr=fashion_x_train)
np.save('./data/fashion_x_test.npy', arr=fashion_x_test)
np.save('./data/fashion_y_train.npy', arr=fashion_y_train)
np.save('./data/fashion_y_test.npy', arr=fashion_y_test)


from tensorflow.keras.datasets import cifar100
(cifar100_x_train, cifar100_y_train), (cifar100_x_test, cifar100_y_test) = cifar100.load_data()
np.save('./data/cifar100_x_train.npy', arr=cifar100_x_train)
np.save('./data/cifar100_x_test.npy', arr=cifar100_x_test)
np.save('./data/cifar100_y_train.npy', arr=cifar100_y_train)
np.save('./data/cifar100_y_test.npy', arr=cifar100_y_test)


from sklearn.datasets import load_boston
boston_datasets = load_boston()
boston_x = boston_datasets.data
boston_y = boston_datasets.target
np.save('./data/boston_x.npy', arr=boston_x)
np.save('./data/boston_y.npy', arr=boston_y)


from sklearn.datasets import load_diabetes
diabetes_datasets = load_diabetes()
diabetes_x = diabetes_datasets.data
diabetes_y = diabetes_datasets.target
np.save('./data/diabetes_x.npy', arr=diabetes_x)
np.save('./data/diabetes_y.npy', arr=diabetes_y)


from sklearn.datasets import load_iris
iris_datasets = load_iris()
iris_x = iris_datasets.data
iris_y = iris_datasets.target
np.save('./data/iris_x.npy', arr=iris_x)
np.save('./data/iris_y.npy', arr=iris_y)


from sklearn.datasets import load_breast_cancer
cancer_datasets = load_breast_cancer()
cancer_x = cancer_datasets.data
cancer_y = cancer_datasets.target
np.save('./data/cancer_x.npy', arr=cancer_x)
np.save('./data/cancer_y.npy', arr=cancer_y)





import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()

x = datasets.data
y = datasets.target
print("x.shape", x.shape) # (442, 10)
print("y.shape", y.shape)


pca = PCA(n_components=7) # n_components = 컬럼 숫자, 목표치

x2d = pca.fit_transform(x)
print(x2d.shape) # (442, 9)

pca_EVR = pca.explained_variance_ratio_
print("pca_EVR:",pca_EVR)
print("sum(pca_EVR):",sum(pca_EVR)) # sum(pca_EVR): 0.9479436357350411





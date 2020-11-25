import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()

x = datasets.data
y = datasets.target
print("x.shape", x.shape) # (442, 10)
print("y.shape", y.shape)


# pca = PCA(n_components=7) # n_components = 컬럼 숫자, 목표치, 10개를 7개로 압축한다

# x2d = pca.fit_transform(x)
# print(x2d.shape) # (442, 9)

# pca_EVR = pca.explained_variance_ratio_
# print("pca_EVR:",pca_EVR)
# print("sum(pca_EVR):",sum(pca_EVR)) # sum(pca_EVR): 0.9479436357350411



pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

print(cumsum)
#    1개 축소 / 2개 축소 / 3개 축소
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]



d = np.argmax(cumsum >= 0.95) +1
print(cumsum>=0.95) 
# [False False False False False False False  True  True  True]
# 0.95를 넘어가는 대표값은 8,9,10 중에 하나

print(d) 
# 8
# 그러니 8을 쓰면 무난 할 것이다


import matplotlib.pyplot as plt
plt.plot(cumsum, marker='.')
plt.grid()
plt.show()









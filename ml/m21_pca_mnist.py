import numpy as np
from tensorflow.keras.datasets import mnist

# y는 사용하지 않을 것이니 _로 표시
(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
print("x.shape:", x.shape) # (70000, 28, 28)


# 실습
# PCA를 통해 0.95이상인게 몇 개 인지 찾으시오
# 그리고 모델을 만드시오

# reshape
x = x.reshape(x.shape[0],x.shape[1]*x.shape[1])
print("after reshape x.shape:", x.shape) # (70000, 784)


from sklearn.decomposition import PCA
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) +1
print(d) # 154



pca = PCA(n_components=d)
x2d = pca.fit_transform(x)
print(x2d.shape) # (70000, 154)

# cumsum 기준을 1로 놓아도, 713이면 된다
# 즉 784 다 쓰는 게, 낭비라는 것


# PCA는 전처리 개념





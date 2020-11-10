# multi layer perceptron

# 1.데이터
import numpy as np
x = np.array((range(1,101),range(711,811), range(100)))
y = np.array((range(101,201), range(311,411), range(100)))

# x = 100행 3열, 데이터 종류 3가지
print(x)
print("before x.shape:", x.shape) # 출력: (3,) array에 range만 3개 저장되어 있다는 뜻

# (100, 3)의 형태로 만들어야 한다

# x = x.reshape(3,100).T
# x = np.transpose(x)
x = x.T
print(x)
print("after x.shape row:", x.shape[0]) # n-dimension
print("after x.shape column:", x.shape[1]) # size of array


# y = y.reshape(3,100).T
# print(y)
# print("after y.shape:", y.shape)





import numpy as np

x = np.array([10,20,30,4])
y = np.array([1,2,3,4])

print(x+y)
print(x-y)
print(x*y)
print(x/y)

b = x==y # [False, False, False, True]
print(b[0])
print(b[1])
print(b[2])
print(b[3])


# 브로드캐스팅
x2 = np.array([[1,2,3,4,5], [1,3,5,7,9]])
y2 = np.ones_like(x2) # x2랑 크기가 같은, 모든 원소 1인 배열
print(y2)
print(x2+5) # 스칼라 5가, 2행5열 배열로 변환되어 연산 됨
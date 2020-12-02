import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

# print(x,"/",y)

plt.plot(x,y)
plt.grid()
plt.show()


# relu 친구들 찾기
# https://yeomko.tistory.com/39
# https://keras.io/ko/layers/advanced-activations/

# relu = REctified Lenear Unit

# elu = 지수 선형 유닛(Exponential Linear Unit) 활성화 함수
#       0이하 값은 exp로 부드럽게 축소(알파값 = 1)

# selu = elu의 알파값이 1이 아닌 경우

# LeakyReLU = relu + 0이하 값은 작은 기울기

# PReLU = Parametric Rectified Linear Unit
#         LeakyReLU + 0이하 값의 기울기 조정 가능

# ThresholdedReLU = 임계값이 있는 ReLU 활성화 함수
#                   relu의 임계값이 0이 아닌 threshold



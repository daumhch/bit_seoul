# 1.데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))

x_train = x[:70] # 70개 
y_train = y[:70]
x_test = x[70:] # 30개
y_test = y[70:] 

print(y_train)
print(y_test)


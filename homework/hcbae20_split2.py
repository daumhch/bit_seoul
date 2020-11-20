import numpy as np
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) -size +1 ):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

def split_x2(seq, size):
    bbb = []
    for i in range(len(seq) - size + 1):
        bbb.append(seq[i:(i+size)])
    return np.array(bbb)

from sklearn.datasets import load_iris
datasets = load_iris()
x1 = split_x(datasets.data, 4)
x2 = split_x2(datasets.data, 4)
print("x1.shape:",x1.shape)
print(x1)
print("x2.shape:",x2.shape)
print(x2)

datasets2 = np.array(range(1,11)) #1부터 10까지 1차원 데이터
print("??????????",type(datasets2))
x3 = split_x(datasets2, 4)
x4 = split_x2(datasets2, 4)
print("x3.shape:",x3.shape)
print("x4.shape:",x4.shape)
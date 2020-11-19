# <<과제>>
# 커스텀 스플릿함수 만든 것의 문제

# iris 데이터를
# 150,4를 5개씩 자른다면 146,5,4가 되어야 한다

# 데이터셋을 자르는 함수 -> 이게 자격증 5번 문제

import numpy as np

def split_x(seq, size):
    aaa = [] # 임시 리스트
    # i는 0부터 seq사이즈-size까지 반복 
    # (그래야 size만큼씩 온전히 자를 수 있다)
    for i in range(len(seq) -size +1 ):
        subset = seq[i:(i+size)] # subset은 i부터 size만큼 배열 저장
        aaa.append([subset]) # 배열에 subset을 붙인다
    print(type(aaa)) # aaa의 타입은 리스트
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자



from sklearn.datasets import load_iris
datasets = load_iris()
x = datasets.data

print("x.shape:", x.shape)

def split_x2(seq, size):
    aaa=[]
    bbb=[]
    for j in range(seq.shape[0]-size+1):
        aaa.clear()
        for i in range(j,j+size):
            aaa.append(seq[i])
        bbb.append(np.array(aaa))
    return np.array(bbb)

x2 = split_x2(x, 5)
print("x2.shape:", x2.shape)

print("x2 = ")
print(x2)

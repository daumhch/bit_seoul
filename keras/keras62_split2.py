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
        aaa.append([item for item in subset]) # 배열에 subset을 붙인다

    print(type(aaa)) # aaa의 타입은 리스트
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자



from sklearn.datasets import load_iris
datasets = load_iris()
x = datasets.data

print("x.shape:", x.shape)

def split_x2(seq, size):
    aaa1=[]
    bbb1=[]
    for j in range(seq.shape[0]-size+1):
        aaa1.clear()
        for i in range(j,j+size):
            aaa1.append(seq[i])
        bbb1.append(np.array(aaa1))
    return np.array(bbb1)

x2 = split_x2(x, 5)
print(x2)
print("x2.shape:", x2.shape)

# print("x2 = ")




# ### 혼용할 수 있는게 만들 수 없을까?
# def split_any(seq, size):
#     ndim = seq.ndim
#     if ndim == 1:
#         return split_x(seq, size)
#     elif ndim == 2:
#         return split_x2(seq, size)
#     else:
#         print('3차원 이상입니다, 아직 구현하지 않았습니다')



# dataset = np.array(range(1,11)) #1부터 10까지 1차원 데이터


# x3 = split_any(x, 5)
# print("x3.shape:",x3.shape)
# print("x3:",x3)

# x4 = split_any(dataset, 3)
# print("x4.shape:",x4.shape)
# print("x4:",x4)

# x5 = split_x(dataset, 3)
# print("x5.shape:",x5.shape)
# print("x5:",x5)







def split_x3(seq, size):
    aaa = []
    print("size :",size)
    for i in range(len(seq) - size + 1):
        aaa.append(seq[i:(i+size)])
    return np.array(aaa)

x6 = split_x3(x, 5)
print(x6)
print(x6.shape)


x7 = split_x(x, 5)
print(x7)
print(x7.shape)

dataset = np.array(range(1,11)) #1부터 10까지 1차원 데이터
x8 = split_x3(dataset, 5)
print(x8)
print(x8.shape)


x9 = split_x(dataset, 5)
print(x9)
print(x9.shape)

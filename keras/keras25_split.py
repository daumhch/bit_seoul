import numpy as np
dataset = np.array(range(1,11)) #1부터 10까지
size = 5 # 5개씩

print(dataset)
# [ 1  2  3  4  5  6  7  8  9 10]


def split_x(seq, size):
    aaa = [] # 임시 리스트
    # i는 0부터 seq사이즈-size까지 반복 
    # (그래야 size만큼씩 온전히 자를 수 있다)
    for i in range(len(seq) -size +1 ):
        subset = seq[i:(i+size)] # subset은 i부터 size만큼 배열 저장
        aaa.append([item for item in subset]) # 배열에 subset을 붙인다

    print(type(aaa)) # aaa의 타입은 리스트
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자

datasets = split_x(dataset, size)
print("================")
print(datasets)
# [ 1  2  3  4  5]
# [ 2  3  4  5  6]
# [ 3  4  5  6  7]
# [ 4  5  6  7  8]
# [ 5  6  7  8  9]
# [ 6  7  8  9 10]



def hcbae_split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        # [item~~~] 이 없어도 되는데, 있는 이유가 있겠지?
        # aaa.insert(i, subset)
        aaa.append(subset)
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자

print("================")
print(hcbae_split_x(dataset, size))




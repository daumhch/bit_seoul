import numpy as np # numpy를 불러오고 앞으로 이름을 np로 쓴다
def hcbae_split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        # [item~~~] 이 없어도 되는데, 있는 이유가 있겠지?
        # aaa.insert(i, subset)
        aaa.append(subset)
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자
# end of hcbae_split_x



import shutil, os
def hcbae_removeAllFile(filePath):
    if os.path.isdir(filePath):
        shutil.rmtree(filePath)
    os.mkdir(os.getcwd()+"\graph")

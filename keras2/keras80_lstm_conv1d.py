# 79 카피
# 실습(embedding 빼고, lstm과 conv1d로 구성)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요','참 최고에요','참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다','한 번 더 보고 싶네요',
        '글쎄요', '별로에요','생각보다 지루해요',
        '연기가 어색해요', '재미없어요','너무 재미없다',
        '참 재밌네요']

# 1:긍정 / 0:부정
labels = np.array([1,1,1,
                    1,1,
                    0,0,0,
                    0,0,0,
                    1])
print(labels.shape)

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생
# 각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24}


x = token.texts_to_sequences(docs)
print(x) # [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24]]

# 근데 길이가 제각각이다
# 짧은 쪽으로 맞추면 데이터가 날아가고
# 긴 쪽으로 맞추면 메모리가 낭비된다
# 빈 곳을 0으로 채우면 길이는 맞출 수 있다
# 만약 뒤를 0으로 채우면 해석의 목표가 0이 되니까,
# 뒷자리 말고 앞자리를 0으로 채워야, 
# 의미있는 데이터가 뒤로가서 해석이 의미가 있다

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre') # pre=앞, post=뒤
print(pad_x)
print(pad_x.shape)
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  1  5  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]]
# (12, 5)

# 25개의 단어를 백터화 한다 (원핫인코딩 대신)

word_size = len(token.word_index) +1
print("전체 토큰 사이즈:", word_size)
# 전체 토큰 사이즈: 25


# reshape
pad_x = pad_x.reshape(pad_x.shape[0],pad_x.shape[1],1)
print("reshape:",pad_x.shape)


# 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, Flatten

model = Sequential()
# model.add(Embedding(word_size, word_size, input_length=pad_x.shape[1] ))
# model.add(LSTM(32, activation='relu', input_shape=(pad_x.shape[1],pad_x.shape[2]) ))
model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(pad_x.shape[1],pad_x.shape[2])) )
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()


# 컴파일 훈련
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['acc'])

model.fit(pad_x, labels, epochs=30)


# 평가 예측
acc = model.evaluate(pad_x, labels)[1]
print("acc:", acc)








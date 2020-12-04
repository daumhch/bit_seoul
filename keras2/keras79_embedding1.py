from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 맛있는 밥을 먹었다'
token = Tokenizer()
token.fit_on_texts([text])
print(token.word_index) # {'나는': 1, '진짜': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}

text = '나는 진짜 맛있는 밥을 진짜 먹었다'
token = Tokenizer()
token.fit_on_texts([text])
print(token.word_index) # {'진짜': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}
# 많이 나오는 단어는 앞으로 간다

x = token.texts_to_sequences([text])
print(x) # [[2, 1, 3, 4, 1, 5]] # 근데 이러면 값을 비교할 수도 있으니, 원핫인코딩

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)+1 # to_categorical은 0부터 시작하니 길이를 +1 한다
x = to_categorical(x, num_classes=word_size)
print(x)








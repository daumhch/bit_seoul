import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

# 정리 참고 사이트
# https://keras.io/ko/layers/convolutional/
# https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/

model = Sequential()
model.add(
    Conv2D(
        10, # filters
        # 정수, 아웃풋 공간의 차원 (다시 말해, 컨볼루션의 아웃풋 필터의 수)
        # 다음 레이어에 던져주는 노드 개수

        (2,2), # kernel_size
        # 정수 혹은 2개 정수의 튜플/리스트. 2D 컨볼루션 윈도우의 높이와 넓이를 특정합니다.
        
        strides=1, # default strides=1
        # 정수 혹은 2개 정수의 튜플/리스트. 
        # 높이와 넓이에 따라 컨볼루션의 보폭 길이를 특정합니다. 
        
        padding='same', # 경계 처리 방법
        # same: 출력 이미지 사이즈가 입력 이미지 사이즈와 동일
        # valid: 유효한 영역만 출력이 됩니다. 출력 이미지 사이즈는 입력 사이즈보다 작습니다
        # 이미지가 5x5인데, kernel_size가 (2,2)라면, 결과가 4x4가 된다
        # same = padding을 씌워서 가장자리의 데이터 손실을 막는다, 결과가 5x5가 된다
        # valid = padding을 적용하지 않는다, 결과가 4x4가 된다

        # 입력모양: batch_size, rows, columns, channels
        # batch_size = 잘라서 연산하는 단위, 행무시
        input_shape=(5,5,1), # (rows, columns, channels)
        # input_shape에서 흑백은 1, 칼라는 3
        
        activation='relu'
        )
    ) 

# 참고 LSTM
# units
# return_sequence
# 입력모양: batch_size, timesteps, feture
# input_shape = (timesteps, feture)

model.summary()




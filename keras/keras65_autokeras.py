# https://autokeras.com/

# Note: Currently, 
# AutoKeras is only compatible with 
# Python >= 3.5 and TensorFlow >= 2.3.0.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기



import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak

# x_train = np.load('./data/keras64_mf_train_x.npy')
# y_train = np.load('./data/keras64_mf_train_y.npy')
# x_test = np.load('./data/keras64_mf_test_x.npy')
# y_test = np.load('./data/keras64_mf_test_y.npy')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(y_train[:3])

# Init image classifier
clf = ak.ImageClassifier(
    # overwrite=True,
    max_trials=2
    )

clf.fit(x_train, y_train, 
    epochs=10
    )

y_predict = clf.predic(x_test)

result = clf.evaluate(x_test, y_test)
print("result:", result)







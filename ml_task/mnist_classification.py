import tensorflow as tf
import numpy as np
import os
import random

np.random.seed(0)
os.environ['PYTHONHASHSEED']=str(0)
random.seed(0)

   
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


def classify(X, Y, x_test, y_test, results=None, temp_results=None):
    
    if temp_results is None:
        temp_results = []
    if results is None:
        results = []
    
    input_shape = (28, 28, 1)

    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))

    X = X.reshape(X.shape[0], 28, 28, 1)

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    max_value = 0
    n_iter = 20
    for _ in range(n_iter):
        model.fit(x=X, y=Y, epochs=1, shuffle=True)
        res = model.evaluate(x_test, y_test)[1]
        if res < max_value:
            temp_results.append(max_value)
            break
        else:
            max_value = res
        
    results.append(temp_results[-1])
        
    return temp_results
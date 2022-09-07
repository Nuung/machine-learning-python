import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import pandas as pd

# neural network를 통한 학습 1 
# Fully connected Neural Net, Eager tensor를 이용
# 비선형 변환은 신경망 처럼 신호가 있고 없고, 0 1 처럼 데이터 처리를 짧게 커트하는 것 처럼 느낄 수 있다 
# 선형 변환 - "비선형 변환" - DEEP NETWORK, 하나하나 layer 쌓아가는 것
# 사실 비선형 변환만 빼면 "행렬의 곱은 선형 변환의 합성이다." 라는 것을 알 수 있다. 
# 그 비선형 변환을 통해 뭘 하느냐

# 우리가 일전에 만들었던 데이터 형태를 생각해보자


p1, p2 = (30,40)
n = 700
x = tf.random.normal((n,p1))
type(x)
h1 = K.layers.Dense(3, activation = 'sigmoid')(x)

z = np.random.normal(size = (n,p2))
h2 = K.layers.Dense(5, activation = 'sigmoid')(z)
h3 = tf.keras.layers.Concatenate(axis = 1)([h1,h2])
h3.shape
h4 = K.layers.Dense(3, activation = 'sigmoid')(h3)

y = np.random.normal(size = (n,3))

# weight ??
s1 = tf.reduce_sum(tf.square(y-h4))
tf.square(y-h4)
s2 = tf.square(y-h4)*tf.constant([1.0, 10.0, 100.0])
tf.reduce_sum(s2)

h1 = K.layers.Dense(3, activation = 'sigmoid')(x)
h2 = K.layers.Dense(5, activation = 'sigmoid')(z)
h3 = tf.keras.layers.Concatenate(axis = 1)([h1,h2])
output = K.layers.Dense(3, activation = 'linear')(h3)



# update가 가능한 model 만들기 (변수들의 이름을 관리함)
# 1. input 정의하기
# 2. output 정의하기
# 3. 모델만들기
input1 = K.layers.Input(p1)
input2 = K.layers.Input(p2)
h1 = K.layers.Dense(3, activation = 'sigmoid')(input1)
h2 = K.layers.Dense(5, activation = 'sigmoid')(input2)
h3 = tf.keras.layers.Concatenate(axis = 1)([h1,h2])
output = K.layers.Dense(3, activation = 'linear')(h3)
my_model = K.models.Model([input1, input2], output)
my_model.summary()
# output evaluation 확인하기
my_model([x,z])


optimizer = K.optimizers.SGD(0.0005)
with tf.GradientTape() as tape:
  yhat = my_model([x,z])
  s2 = tf.square(y-yhat)*tf.constant([0.1, 0.2, 0.7])
  loss = tf.reduce_sum(s2)
  
grad = tape.gradient(loss, my_model.trainable_weights)
optimizer.apply_gradients(zip(grad, my_model.trainable_weights))

# estimation
for i in range(1000):
    with tf.GradientTape() as tape:
        yhat = my_model([x,z])
        s2 = tf.square(y-yhat)*tf.constant([0.1, 0.2, 0.7])
        loss = tf.reduce_sum(s2)
    grad = tape.gradient(loss, my_model.trainable_weights)
    optimizer.apply_gradients(zip(grad, my_model.trainable_weights))
    if (i%10==0):
        print(loss,'\n')
 


    

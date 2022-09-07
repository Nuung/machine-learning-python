import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tmp_y = pd.read_csv("./real/y.csv")
tmp_x = pd.read_csv("./real/x.csv")
tmp_name = list(tmp_y.columns)

tmp_y.shape
tmp_x.shape
# x: (671, 455), y: (671,424) 
x = np.array(tmp_x)
y = np.array(tmp_y)
district_name = [str(tmp_name[i])[1:] for i in range(y.shape[1])]

p = x.shape[1]; q = y.shape[1]
p1 = y.shape[1]
p2 = x.shape[1] - y.shape[1]
n = x.shape[0]


x1 = x[:,:p1]
x2 = x[:, p1:]
x1.shape
x2.shape
# type 1
# update가 가능한 model 만들기 (변수들의 이름을 관리함)
# 1. input 정의하기
# 2. output 정의하기
# 3. 모델만들기

input1 = K.layers.Input(p1)
input2 = K.layers.Input(p2)
h1 = K.layers.Dense(10, activation = 'sigmoid')(input1)
h2 = K.layers.Dense(5, activation = 'sigmoid')(input2)
h3 = tf.keras.layers.Concatenate(axis = 1)([h1,h2])
output = K.layers.Dense(p1, activation = 'linear')(h3)
my_model = K.models.Model([input1, input2], output)
my_model.summary()
# output evaluation 확인하기
my_model([x1,x2])


optimizer = K.optimizers.SGD(0.1)
with tf.GradientTape() as tape:
  yhat = my_model([x1,x2])
  s2 = tf.square(y-yhat)
  loss = tf.reduce_sum(s2)
  
grad = tape.gradient(loss, my_model.trainable_weights)
optimizer.apply_gradients(zip(grad, my_model.trainable_weights))

optimizer = K.optimizers.SGD(0.5)
# estimation
for i in range(10000):
    with tf.GradientTape() as tape:
        yhat = my_model([x1,x2])
        s2 = tf.square(y-yhat)
        loss = tf.reduce_mean(s2)
    grad = tape.gradient(loss, my_model.trainable_weights)
    optimizer.apply_gradients(zip(grad, my_model.trainable_weights))
    if (i%100==0):
        print(loss,'\n')
 
v = my_model([x1,x2])
err = np.mean(y-v.numpy(), axis = 0)
err.mean()
plt.plot(err)
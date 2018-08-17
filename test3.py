# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 18:42:04 2018

@author: schabc
"""

import tensorflow as tf
import numpy as np

# creat data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y_pre = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y_pre - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5) #优化器
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# create tensorflow structure end ###

sess = tf.Session()
sess.run(init)  # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights), sess.run(biases))
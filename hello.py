# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 11:24:12 2018

@author: schabc
"""

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')  
sess = tf.Session()  
print(sess.run(hello)) 
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 20:33:04 2018

@author: schabc
"""

from datetime import datetime
import math
import time
import tensorflow as tf

#卷积层函数
def conv_op(input_op, name, kernelheight, kernelwidth, n_out, dh, dw, para):
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[kernelheight,kernelwidth,n_in,n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        
        conv = tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val,trainable=True,name='b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z, name=scope)
        para += [kernel, biases]
        return activation
    
#全连接层函数
def fc_op(input_op, name, n_out, para):
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                 shape=[n_in,n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1,
                                         shape=[n_out],
                                         dtype=tf.float32),
                             name='b')
        activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        para += [kernel, biases]
        return activation
    
#最大池化层函数
def mpool_op(input_op,name,kernelheight, kernelwidth, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1,kernelheight,kernelwidth,1],
                          strides=[1,dh,dw,1],
                          padding='SAME',
                          name=name)
    
#VGGNet网络结构
def inference_op(input_op, keep_prob):
    para = []
    
    #A类卷积层
    conv1_1 = conv_op(input_op,name="conv1_1",kernelheight=3,kernelwidth=3,n_out=64,dh=1,dw=1,para=para)
    conv1_2 = conv_op(conv1_1,name="conv1_2",kernelheight=3,kernelwidth=3,n_out=64,dh=1,dw=1,para=para)
    pool1 = mpool_op(conv1_2,name="pool1",kernelheight=2,kernelwidth=2,dh=2,dw=2)
    
    #B类卷积层
    conv2_1 = conv_op(pool1,name="conv2_1",kernelheight=3,kernelwidth=3,n_out=128,dh=1,dw=1,para=para)
    conv2_2 = conv_op(conv2_1,name="conv2_2",kernelheight=3,kernelwidth=3,n_out=128,dh=1,dw=1,para=para)
    pool2 = mpool_op(conv2_2,name="pool2",kernelheight=2,kernelwidth=2,dh=2,dw=2)
    
    #C类卷积层
    conv3_1 = conv_op(pool2,name="conv3_1",kernelheight=3,kernelwidth=3,n_out=256,dh=1,dw=1,para=para)
    conv3_2 = conv_op(conv3_1,name="conv3_2",kernelheight=3,kernelwidth=3,n_out=256,dh=1,dw=1,para=para)
    conv3_3 = conv_op(conv3_2,name="conv3_3",kernelheight=3,kernelwidth=3,n_out=256,dh=1,dw=1,para=para)
    pool3 = mpool_op(conv3_3,name="pool3",kernelheight=2,kernelwidth=2,dh=2,dw=2)
    
    #D类卷积层
    conv4_1 = conv_op(pool3,name="conv4_1",kernelheight=3,kernelwidth=3,n_out=512,dh=1,dw=1,para=para)
    conv4_2 = conv_op(conv4_1,name="conv4_2",kernelheight=3,kernelwidth=3,n_out=512,dh=1,dw=1,para=para)
    conv4_3 = conv_op(conv4_2,name="conv4_3",kernelheight=3,kernelwidth=3,n_out=512,dh=1,dw=1,para=para)
    pool4 = mpool_op(conv4_3,name="pool4",kernelheight=2,kernelwidth=2,dh=2,dw=2)
        
    #F类卷积层
    conv5_1 = conv_op(pool4,name="conv5_1",kernelheight=3,kernelwidth=3,n_out=512,dh=1,dw=1,para=para)
    conv5_2 = conv_op(conv5_1,name="conv5_2",kernelheight=3,kernelwidth=3,n_out=512,dh=1,dw=1,para=para)
    conv5_3 = conv_op(conv5_2,name="conv5_3",kernelheight=3,kernelwidth=3,n_out=512,dh=1,dw=1,para=para)
    pool5 = mpool_op(conv5_3,name="pool5",kernelheight=2,kernelwidth=2,dh=2,dw=2)
    
    #转换数据shape
    shape = pool5.get_shape()
    flattened_shape = shape[1].value * shape[2].value * shape[3].value
    resh1 = tf.reshape(pool5,[-1, flattened_shape],name="resh1")
    
    #全连接层1
    fc6 = fc_op(resh1,name="fc6",n_out=4096,para=para)
    fc6_drop = tf.nn.dropout(fc6,keep_prob,name="fc6_drop")
    
    #全连接层2
    fc7 = fc_op(fc6_drop,name="fc7",n_out=4096,para=para)
    fc7_drop = tf.nn.dropout(fc7,keep_prob,name="fc7_drop")
    
    #输出层
    fc8 = fc_op(fc7_drop,name="fc8",n_out=1000,para=para)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax,1)
    return predictions, softmax, fc8, para

def time_tensorflow_run(session,target,feed,info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target,feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' %
                       (datetime.now(),i-num_steps_burn_in,duration))
                total_duration += duration
                total_duration_squared += duration*duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn*mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))
    
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, para = inference_op(images, keep_prob)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess,predictions,{keep_prob:1.0},"Forward")
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, para)
        time_tensorflow_run(sess,grad,{keep_prob:0.5},"Forward-backward")
        
batch_size = 32
num_batches = 100
run_benchmark()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
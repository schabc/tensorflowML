# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 11:23:34 2018

@author: schabc
"""

import tensorflow as tf
#import numpy as np
import matplotlib.pyplot as plt
import time
import cifar10_input,cifar10    
#用于加载和处理数据,来自tensorflow/models/tutotials/image/cifar10

max_steps = 3000   #一共训练的两万多次，分了两次，中途保存过一次参数变量
batch_size = 256    # 小批量数据大小
s_times =20         # 每轮训练数据的组数，每组为一batchsize

learning_rate = 0.001
data_dir = 'data/cifar-10-batches-bin/'    # 数据所在路径

#cifar10.data_path = "data/cifar-10-python/cifar-10-batches-bin/"
#cifar10.maybe_download_and_extract() 

# Xavier初始化方法
# 卷积权重(核）初始化
def init_conv_weights(shape, name):
    weights = tf.get_variable(name=name, shape=shape, dtype=tf.float32, 
                             initializer=tf.contrib.layers.xavier_initializer_conv2d())
    return weights

# 全连接权重初始化
def init_fc_weights(shape, name):
    weights = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
    return weights

# 偏置
def init_biases(shape, name):
    biases = tf.Variable(tf.random_normal(shape),name=name, dtype=tf.float32)
    return biases


# 卷积
# 参数：输入张量,卷积核，偏置，卷积核在高和宽维度上移动的步长
def conv2d(input_tensor, weights, biases, s_h, s_w):
    conv = tf.nn.conv2d(input_tensor, weights, [1, s_h, s_w, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

# 池化
# 参数：输入张量，池化核高和宽，池化核在高，宽维度上移动步长
def max_pool(input_tensor, k_h, k_w, s_h, s_w):
    return tf.nn.max_pool(input_tensor, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='SAME')

# 全链接
# 参数：输入张量，全连接权重，偏置
def fullc(input_tensor, weights, biases):
    return tf.nn.relu_layer(input_tensor, weights, biases)


# 输入占位节点
images = tf.placeholder(tf.float32, [batch_size, 24 ,24 ,3])
labels = tf.placeholder(tf.int32, [batch_size])
# 正则
keep_prob = tf.placeholder(tf.float32)

# 使用作用域对Ｏｐ进行封装，在使用tensorboard对网络结构进行可视化的效果较好

# 第一组卷积 conv3-16
with tf.name_scope('conv_group_1'):
    cw1 = init_conv_weights([3, 3, 3, 16], name='conv_w1')
    cb1 = init_biases([16], name='conv_b1')
    conv1 = conv2d(images, cw1, cb1, 1, 1)

# 最大池化 2x2
pool1 = max_pool(conv1, 2, 2, 2, 2)


# 第二组卷积　conv3-32
with tf.name_scope('conv_group_2'):
    cw2 = init_conv_weights([3, 3, 16, 32], name='conv_w2')
    cb2 = init_biases([32], name='conv_b2')
    conv2 = conv2d(pool1, cw2, cb2, 1, 1)

# 最大池化
pool2 = max_pool(conv2, 2, 2, 2, 2)

# 第三组卷积　conv3-64  conv3-64
with tf.name_scope('conv_group_3'):
    cw3 = init_conv_weights([3, 3, 32, 64], name='conv_w3')
    cb3 = init_biases([64], name='conv_b3')
    conv3 = conv2d(pool2, cw3, cb3, 1, 1)

    cw4 = init_conv_weights([3, 3, 64, 64], name='conv_w4')
    cb4 = init_biases([64], name='conv_b4')
    conv4 = conv2d(conv3, cw4, cb4, 1, 1)

# 最大池化
pool3 = max_pool(conv4, 2, 2, 2, 2)     

# 第四组卷积　conv3-128 conv3-128
with tf.name_scope('conv_group_4'):
    cw5 = init_conv_weights([3, 3, 64, 128], name='conv_w5')
    cb5 = init_biases([128], name='conv_b5')
    conv5 = conv2d(pool3, cw5, cb5, 1, 1)

    cw6 = init_conv_weights([3, 3, 128, 128], name='conv_w6')
    cb6 = init_biases([128], name='conv_b6')
    conv6 = conv2d(conv5, cw6, cb6, 1, 1)

# 此时张量的高和宽为　３ｘ３，继续池化为　２ｘ２
#pool4 = max_pool(conv6, 2, 2, 2, 2)

# 第五组卷积　conv3-256 conv3-256
with tf.name_scope('conv_group_5'):
    cw7 = init_conv_weights([3, 3, 128, 128], name='conv_w7')
    cb7 = init_biases([128], name='conv_b7')
    conv7 = conv2d(conv6, cw7, cb7, 1, 1)

    cw8 = init_conv_weights([3, 3, 128, 128], name='conv_w8')
    cb8 = init_biases([128], name='conv_b8')
    conv8 = conv2d(conv7, cw8, cb8, 1, 1)

# 此处应该还有一个池化，但是现在张量为３ｘ３，很小了，所以省略了池化
    
# 转换数据shape
reshape_conv8 = tf.reshape(conv8, [batch_size, -1])
n_in = reshape_conv8.get_shape()[-1].value
    
# 地一个全连接层命名空间
with tf.name_scope('fullc_1'):
    fw9 = init_fc_weights([n_in, 256], name='fullc_w9')
    fb9 = init_biases([256], name='fullc_b9')
    activation1 = fullc(reshape_conv8, fw9, fb9)

# dropout正则
drop_act1 = tf.nn.dropout(activation1, keep_prob)

with tf.name_scope('fullc_2'):
    fw10 = init_fc_weights([256, 256], name='fullc_w10')
    fb10 = init_biases([256], name='fullc_b10')
    activation2 = fullc(drop_act1, fw10, fb10)

# dropout正则
drop_act2 = tf.nn.dropout(activation2, keep_prob)

with tf.name_scope('fullc_3'):
    fw11 = init_fc_weights([256, 10], name='fullc_w11')
    fb11 = init_biases([10], name='full_b11')
    logits = tf.add(tf.matmul(drop_act2, fw11), fb11)
    output = tf.nn.softmax(logits)
    
    
#损失函数和优化器
cross_entropy= tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits, labels=labels)
cost = tf.reduce_mean(cross_entropy,name='Train_Cost')
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 用来评估测试数据的准确率
# 数据labels没有使用one-hot编码格式,labels是int32
def accuracy(labels, output):
    labels = tf.to_int64(labels)
    pred_result = tf.equal(labels, tf.argmax(output, 1))
    accu = tf.reduce_mean(tf.cast(pred_result, tf.float32))
    return accu

# 加载训练batch_size大小的数据，经过增强处理，剪裁，反转，等等
train_images, train_labels = cifar10_input.distorted_inputs(batch_size= batch_size, data_dir= data_dir)

#Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
# 加载测试数据，batch_size大小，不进行增强处理
test_images, test_labels = cifar10_input.inputs(batch_size= batch_size, data_dir= data_dir,eval_data= True)


# Training
def training(sess, max_steps, s_times, keeprob, display):

    Cost = []
    for i in range(max_steps):
        for j in range(s_times):
            start = time.time()
            batch_images, batch_labels = sess.run([train_images, train_labels])
            opt = sess.run(optimizer, feed_dict={images:batch_images, labels:batch_labels,
                                                keep_prob:keeprob})
            every_batch_time = time.time() - start
        c = sess.run(cost, feed_dict={images:batch_images, labels:batch_labels,
                                        keep_prob:keeprob})

        Cost.append(c)
        if i % display == 0:
            samples_per_sec = float(batch_size) / every_batch_time
            format_str = 'Epoch %d: %d samples/sec, %.4f sec/batch, Cost : %.5f'
            print (format_str%(i+display, samples_per_sec, every_batch_time, c))

    return Cost

#会话
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 图片增强处理,时使用了16个线程加速,启动16个独立线程
tf.train.start_queue_runners(sess=sess)

















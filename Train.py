# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 20:33:08 2018

@author: schabc
"""

import tensorflow as tf
import ReadData
import InceResNetV2 as InReV2
import numpy as np
slim = tf.contrib.slim

########定义函数生成网络中经常用到的函数的默认参数########
# 默认参数：卷积的激活函数、权重初始化方式、标准化器等
def arg_scope(weight_decay=0.00004,  # 设置L2正则的weight_decay
                           stddev=0.1, # 标准差默认值0.1
                           batch_norm_var_collection='moving_vars'):

  batch_norm_params = {  # 定义batch normalization（标准化）的参数字典
      'decay': 0.9997,  # 定义参数衰减系数
      'epsilon': 0.001,  
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }

  with slim.arg_scope([slim.conv2d, slim.fully_connected], # 给函数的参数自动赋予某些默认值
                      weights_regularizer=slim.l2_regularizer(weight_decay)): # 对[slim.conv2d, slim.fully_connected]自动赋值
  # 使用slim.arg_scope后就不需要每次都重复设置参数了，只需要在有修改时设置
    with slim.arg_scope( # 嵌套一个slim.arg_scope对卷积层生成函数slim.conv2d的几个参数赋予默认值
        [slim.conv2d],
        weights_initializer=tf.truncated_normal_initializer(0.0, stddev), # 权重初始化器
        activation_fn=tf.nn.relu, # 激活函数
        normalizer_fn=slim.batch_norm, # 标准化器
        normalizer_params=batch_norm_params) as sc: # 标准化器的参数设置为前面定义的batch_norm_params
      return sc # 最后返回定义好的scope
  
def arg_scope1():
    with slim.arg_scope([slim.conv2d, slim.fully_connected],  
                      activation_fn=tf.nn.relu,  
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),  
                      weights_regularizer=slim.l2_regularizer(0.0005)) as sc:
        return sc
    
# 通过TensorFlow-Slim来定义LeNet-5的网络结构。
def lenet5(inputs, num_classes):
    net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='layer1-conv')
    net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')
    net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='layer3-conv')
    net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, 500, scope='layer5')
    net = slim.fully_connected(net, num_classes, scope='output')
    return net

#定义一个vgg16网络 网络宽度[64,128,256,512,512,4096,4096][16,32,64,128,128,500,500]
def vgg16(inputs, num_classes):       
    net = slim.repeat(inputs, 2, slim.conv2d, 16, [3, 3], scope='conv1')  
    net = slim.max_pool2d(net, [2, 2], scope='pool1')  
    net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='conv2')  
    net = slim.max_pool2d(net, [2, 2], scope='pool2')  
    net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='conv3')  
    net = slim.max_pool2d(net, [2, 2], scope='pool3')  
    net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv4')  
    net = slim.max_pool2d(net, [2, 2], scope='pool4')  
    net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv5')  
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, 500, scope='fc6')  
    net = slim.dropout(net, 0.5, scope='dropout6')  
    net = slim.fully_connected(net, 500, scope='fc7')  
    net = slim.dropout(net, 0.5, scope='dropout7')  
    net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc8')  
    return net 
    
def train(train,test,channels = 3):
    # 训练数据 及 标签
    data_size = train.images[0].shape[0]
    label_size = train.labels[0].shape[0]
    x = tf.placeholder(tf.float32, [None, data_size], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, label_size], name='y-input')
    # 对数据进行训练
    image_size = int(np.sqrt(data_size/channels))
    inputs = tf.reshape(x, [-1, image_size, image_size, channels])
    with slim.arg_scope(arg_scope()):
        #y = InReV2.InceResNetV2(inputs,label_size)
        y = vgg16(inputs,label_size)

    # 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
    # 计算损失
    loss = tf.reduce_mean(cross_entropy)
    # 优化
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    #预测结果评估        
    correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(10):
            for i in range(501):
                xs, ys = train.next_batch(100)
                _, loss_value = sess.run([train_op, loss], feed_dict={x: xs, y_: ys})
    
                if i % 100 == 0 and i != 0:
                    print("After %d training step(s), loss on training batch is %g." % (i, loss_value))
            accuracy_value = sess.run([accuracy],feed_dict ={x: test.images, y_: test.labels})
            print ('epoch ', epoch+1, "; acc" , accuracy_value)
        
# 3. 主程序
def main(argv=None):
    traindata, testdata = ReadData.load_CIFAR10()
    train(traindata, testdata)

if __name__ == "__main__":
    main()
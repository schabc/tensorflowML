# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:26:38 2018

@author: schabc
"""

import tensorflow as tf
slim = tf.contrib.slim

def Stem(inputs):
    output = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='VALID')
    output = slim.conv2d(output, 32, [3, 3], padding='VALID')
    output = slim.conv2d(output, 64, [3, 3])
    output_left = slim.max_pool2d(output, [3, 3])
    output_right = slim.conv2d(output, 96, [3, 3], stride=2, padding='VALID')
    output = tf.concat([output_left, output_right], 3)

    output_left = slim.conv2d(output, 64, [1, 1])
    output_left = slim.conv2d(output_left, 96, [3, 3], padding='VALID')
    output_right = slim.conv2d(output, 64, [1, 1])
    output_right = slim.conv2d(output_right, 64, [7, 1])
    output_right = slim.conv2d(output_right, 64, [1, 7])
    output_right = slim.conv2d(output_right, 96, [3, 3], padding='VALID')
    output = tf.concat([output_left, output_right], 3)

    output_left = slim.conv2d(output, 192, [3, 3], stride=2, padding='VALID')
    output_right = slim.max_pool2d(output, [3, 3])
    output = tf.concat([output_left, output_right], 3)

    return tf.nn.relu(output)


def Inception_ResNet_A(inputs, activation_fn=tf.nn.relu):
    output_res = tf.identity(inputs)

    output_inception_a = slim.conv2d(inputs, 32, [1, 1])
    output_inception_a = slim.conv2d(output_inception_a, 384, [1, 1], activation_fn=None) 

    output_inception_b = slim.conv2d(inputs, 32, [1, 1])
    output_inception_b = slim.conv2d(output_inception_b, 32, [3, 3])
    output_inception_b = slim.conv2d(output_inception_b, 384, [1, 1], activation_fn=None)

    output_inception_c = slim.conv2d(inputs , 32, [1, 1])
    output_inception_c = slim.conv2d(output_inception_c, 48, [3, 3])
    output_inception_c = slim.conv2d(output_inception_c, 64, [3, 3])
    output_inception_c = slim.conv2d(output_inception_c, 384, [1, 1], activation_fn=None)

    output_inception = tf.add_n([output_inception_a, output_inception_b, output_inception_c])
    output_inception = tf.multiply(output_inception, 0.1)

    return activation_fn(tf.add_n([output_res, output_inception]))

 
def Reduction_A(inputs):
    output_a = slim.max_pool2d(inputs, [3, 3])

    output_b = slim.conv2d(inputs, 384, [3, 3], stride=2, padding='VALID')

    output_c = slim.conv2d(inputs, 256, [1, 1])
    output_c = slim.conv2d(output_c, 256, [3, 3])
    output_c = slim.conv2d(output_c, 384, [3, 3], stride=2, padding='VALID')

    return tf.nn.relu(tf.concat([output_a, output_b, output_c], 3))


def Inception_ResNet_B(inputs, acfivation_fn=tf.nn.relu):
    output_res = tf.identity(inputs)

    output_a = slim.conv2d(inputs, 192, [1, 1])
    output_a = slim.conv2d(output_a, 1152, [1, 1], activation_fn=None)

    output_b = slim.conv2d(inputs, 128, [1, 1])
    output_b = slim.conv2d(output_b, 160, [1, 7])
    output_b = slim.conv2d(output_b, 192, [7, 1])
    output_b = slim.conv2d(output_b, 1152, [1, 1], activation_fn=None)

    output = tf.add_n([output_a, output_b])

    output = tf.multiply(output, 0.1)

    return acfivation_fn(tf.add_n([output_res, output]))



def Reduction_B(inputs):
    output_a = slim.max_pool2d(inputs, [3, 3])

    output_b = slim.conv2d(inputs, 256, [1, 1])
    output_b = slim.conv2d(output_b, 384, [3, 3], stride=2, padding='VALID')

    output_c = slim.conv2d(inputs, 256, [1, 1])
    output_c = slim.conv2d(output_c, 256, [1, 1])
    output_c = slim.conv2d(output_c, 288, [3, 3], stride=2, padding='VALID')

    output_d = slim.conv2d(inputs, 256, [1, 1])
    output_d = slim.conv2d(output_d, 288, [3, 3])
    output_d = slim.conv2d(output_d, 320, [3, 3], stride=2, padding='VALID')

    return tf.nn.relu(tf.concat([output_a, output_b, output_c, output_d], 3))   


def Inception_ResNet_C(inputs, activation_fn=tf.nn.relu):
    output_res = tf.identity(inputs) 

    output_a = slim.conv2d(inputs, 192, [1, 1])
    output_a = slim.conv2d(output_a, 2144, [1, 1], activation_fn=None)

    output_b = slim.conv2d(inputs, 192, [1, 1])
    output_b = slim.conv2d(output_b, 224, [1, 3])
    output_b = slim.conv2d(output_b, 256, [3, 1])
    output_b = slim.conv2d(output_b, 2144, [1, 1], activation_fn=None)

    output = tf.add_n([output_a, output_b])

    output = tf.multiply(output, 0.1)

    return activation_fn(tf.add_n([output_res, output]))


def Average_Pooling(inputs):
    output = slim.avg_pool2d(inputs, [8, 8])
    return output
 
def Dropout(inputs, keep=0.8):
    output = slim.dropout(inputs, keep_prob=keep)
    return output


def InceResNetV2(inputs, num_classes):
#    with slim.arg_scope([slim.conv2d],
#                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
#                        activation_fn = tf.nn.relu,
#                        normalizer_fn = slim.batch_norm):

        with tf.name_scope('Stem'):
            output = Stem(inputs)

        with tf.name_scope('5xInception-ResNet-A'):
            for i in range(5):
                output = Inception_ResNet_A(output) 

        with tf.name_scope('Reduction-A'):
            output = Reduction_A(output)

        with tf.name_scope('10xInception-ResNet-B'):
            for i in range(10):
                output = Inception_ResNet_B(output)

        with tf.name_scope('Reduction-B'):
            output = Reduction_B(output)

        with tf.name_scope('5xInception-ResNet-C'):
            for i in range(5):
                output = Inception_ResNet_C(output)

        with tf.name_scope('AveragePooling'):
            output = Average_Pooling(output)

        with tf.name_scope('Dropout0.8'):
            output = Dropout(output)
            output = slim.flatten(output) 

        with tf.name_scope('fc'):
            output = slim.fully_connected(output,num_classes)
            
        output = slim.flatten(output, scope='flatten')
        return output

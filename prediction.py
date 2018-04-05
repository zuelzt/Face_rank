#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 10:03:48 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')


import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# parameters
learning_rate = 0.001
batch_size = 20
display_step = 5

# nn parameters
n_class = 10  # rank 1 - 10
dropout = 0.75

# set
with tf.name_scope('Placeholder'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 128 ,128 ,3])
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_class])
    keep_prob = tf.placeholder(tf.float32)  # keep probability

# create model
def conv2d(X, w, b, strides=1):
    x = tf.nn.conv2d(X, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(X, k=2):
    return tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def cnn(X, weights, biases, dropout):
    x = tf.reshape(X, shape=[-1, 128, 128, 3])
    # layer1
    conv_1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv_1 = maxpool2d(conv_1, k=2)
    tf.summary.histogram('relu1', conv_1)
    # layer2
    conv_2 = conv2d(conv_1, weights['wc2'], biases['bc2'])
    conv_2 = maxpool2d(conv_2, k=2)
    tf.summary.histogram('relu2', conv_2)
    # fully connected layer
    fc = tf.reshape(conv_2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, weights['wd1']), biases['bd1'])
    fc = tf.nn.relu(fc)
    tf.summary.histogram('relu3', fc)
    #drop out
    do = tf.nn.dropout(fc, dropout)
    tf.summary.histogram('dropout', do)
    out_layer = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    return out_layer
    
# weights, biases
weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, 24])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 24, 96])),
        'wd1': tf.Variable(tf.random_normal([32*32*96, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_class]))
        }

biases = {
        'bc1': tf.Variable(tf.random_normal([24])),
        'bc2': tf.Variable(tf.random_normal([96])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_class]))
        }

# construct model
with tf.name_scope('Model'):
    pred = cnn(X, weights, biases, keep_prob)
    result = tf.argmax(pred, 1)

with tf.name_scope('Cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    
with tf.name_scope('GD'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initialize
init = tf.global_variables_initializer()
saver = tf.train.Saver()  # store model


# set path
path = '/Users/zt/Desktop/face/test2/face_ex/zt/face_ex' + '/'

# predict
with tf.Session() as sess:
    saver.restore(sess, '/Users/zt/Desktop/model.ckpt')  # reload model
    image_list = os.listdir(path)
    if '.DS_Store' in image_list:
        image_list.remove('.DS_Store')
    x_in = []
    name = []
    begin = time.time()
    for i in range(len(image_list)):  # import test
        image_name = image_list[i]
        image_name = path + image_name
        img = Image.open(image_name)
        img_array = np.asarray(img, dtype='float32')
        image = np.reshape(img_array, [128, 128, 3])
        x_in.append(image)
        name.append(image_name)
    x_in = np.asarray(x_in)  # set input data
    # result
    prediction = sess.run(result, feed_dict={X: x_in,
                                             keep_prob: 1.})
    for i in range(len(prediction)):
        os.rename(name[i],
                  path + str(prediction[i]) + '----' + image_list[i])
    end = time.time()
    print('-------------------------------------')
    print('Your mean_score is {0:.1f}'.format(np.mean(prediction)))
    print('Done! Total {0:.3f} s'.format(end - begin))
    print('-------------------------------------')

    
    
    






















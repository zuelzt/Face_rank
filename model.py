#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:25:14 2018

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
import random

# parameters
learning_rate = 0.001
batch_size = 10
display_step = 5

# nn parameters
n_class = 10  # rank 1 - 10
dropout = 0.75

# log
log_path = '/Users/zt/Desktop/TensorFlow/face/logs/'

# set
with tf.name_scope('Placeholder'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 128 ,128 ,3], name='X')
    tf.summary.image('Input', X)
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_class], name='y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # keep probability

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
    tf.summary.histogram('wc1', weights['wc1'])
    tf.summary.histogram('bc1', biases['bc1'])
    # layer2
    conv_2 = conv2d(conv_1, weights['wc2'], biases['bc2'])
    conv_2 = maxpool2d(conv_2, k=2)
    tf.summary.histogram('relu2', conv_2)
    tf.summary.histogram('wc2', weights['wc2'])
    tf.summary.histogram('bc2', biases['bc2'])
    # fully connected layer
    fc = tf.reshape(conv_2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, weights['wd1']), biases['bd1'])
    fc = tf.nn.relu(fc)
    tf.summary.histogram('relu3', fc)
    tf.summary.histogram('wd1', weights['wd1'])
    tf.summary.histogram('bd1', biases['bd1'])
    #drop out
    do = tf.nn.dropout(fc, dropout)
    tf.summary.histogram('dropout', do)
    out_layer = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    tf.summary.histogram('w_out', weights['out'])
    tf.summary.histogram('b_out', biases['out'])
    return out_layer
    
# weights, biases
with tf.name_scope('weights'):
    weights = {
            'wc1': tf.Variable(tf.random_normal([5, 5, 3, 24]), name='wc1'),
            'wc2': tf.Variable(tf.random_normal([5, 5, 24, 96]), name='wc2'),
            'wd1': tf.Variable(tf.random_normal([32*32*96, 1024]), name='wd1'),
            'out': tf.Variable(tf.random_normal([1024, n_class]), name='out')
            }
with tf.name_scope('biases'):
    biases = {
            'bc1': tf.Variable(tf.random_normal([24]), name='bc1'),
            'bc2': tf.Variable(tf.random_normal([96]), name='bc2'),
            'bd1': tf.Variable(tf.random_normal([1024]), name='bd2'),
            'out': tf.Variable(tf.random_normal([n_class]), name='out')
            }

# construct model
with tf.name_scope('Model'):
    pred = cnn(X, weights, biases, keep_prob)

with tf.name_scope('Cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    
with tf.name_scope('Adam'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initialize
init = tf.global_variables_initializer()
saver = tf.train.Saver()  # store model
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

# run
with tf.Session() as sess:
    sess.run(init)
    step = 1
    image_list = os.listdir('/Users/zt/Desktop/TensorFlow/face/data/face_ex')
    count = 0
    writer = tf.summary.FileWriter(log_path, graph=sess.graph)
    begin = time.time()
    
    while count < 25:  # 重复迭代 25 次
        random.shuffle(image_list)
        count += 1
        
        for batch_id in range(0,25):
            batch = image_list[batch_id * batch_size:batch_id * batch_size + batch_size]
            # [0:20], [20:40], .... , [480:500]
            batch_xs = []
            batch_ys = []
            for image in batch:  # 每张图提取
                location = image.find('-')
                image_id = image[:location]
                score = image[location + 1:location + 2]
                # open image
                img = Image.open('/Users/zt/Desktop/TensorFlow/face/data/face_ex/' + image)
                # batch_x
                img = np.asarray(img, dtype='float32')
                batch_x = np.reshape(img, [128, 128, 3])
                batch_xs.append(batch_x)
                # batch_y
                batch_y = np.repeat(0, 10)
                batch_y[int(score) - 1] = 1
                batch_y = np.reshape(batch_y, [10, ])
                batch_ys.append(batch_y)
            batch_xs = np.asarray(batch_xs)
            batch_ys = np.asarray(batch_ys)
            #train
            sess.run(optimizer, feed_dict={X: batch_xs,
                                           y: batch_ys,
                                           keep_prob: dropout})
            if step % display_step == 0:
                loss, acc, summary = sess.run([cost, accuracy, merged], feed_dict={X: batch_xs,
                                     y: batch_ys,
                                     keep_prob: 1.})
                writer.add_summary(summary, count * batch_id + 1)
                print('Step {0} -- loss: {1:.3f} -- Accuracy: {2:.2f}'
                      .format(step, loss, acc))
            step += 1 
        SSS = sess.run(merged, feed_dict={X: batch_xs,
                                     y: batch_ys,
                                     keep_prob: 1.})
        writer.add_summary(summary, count)
    end = time.time()
    print('Finished! Total {0:.3f} s.'.format(end - begin))
    saver.save(sess, '/Users/zt/Desktop/model.ckpt')










































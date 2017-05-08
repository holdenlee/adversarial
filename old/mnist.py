from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def inference(x, keep_prob):
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1,28,28,1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv

def loss(logits, labels):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', cross_entropy)
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate=1e-4):
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def simple_fs(x, y_):
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  # Define loss and optimizer
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return cross_entropy, train_step, accuracy

"""
if __name__=='__main__':
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  cross_entropy, train_step, accuracy = simple_fs(x, y_)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    #print(batch[0])
    #print(batch[1])
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1]}, session=sess)
      print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]}, session=sess)

  print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}, session=sess))
"""

if __name__=="__main__":
  # read the data
  #mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  #batch = mnist.train.next_batch(4)
  #print(batch[0])
  #print(batch[1])
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
  #batch = mnist.train.next_batch(4)
  #print(batch[0])
  #print(batch[1])
  #input variable
  x = tf.placeholder(tf.float32, [None, 784])
  keep_prob = tf.placeholder(tf.float32)
  y = inference(x, keep_prob)
  #label
  y_ = tf.placeholder(tf.int64, [None]) #,10
  #loss
  cross_entropy = loss(y, y_)
  train_step = training(cross_entropy, 1e-4) #1e-4)
  correct_prediction = tf.equal(tf.argmax(y,1), y_)
#tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    #print(batch[0])
    #print(batch[1])
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0}, session=sess)
      print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, session=sess)

  print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}, session=sess))


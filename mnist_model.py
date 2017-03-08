# Use the default MNIST model which gets 99% accuracy

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

def conv_and_pool(x, m, fdim, ksize, scope="conv_and_pool"):
    #note by default this includes a relu
    with tf.name_scope(scope) as scope:
        net = slim.conv2d(x, m, fdim)
        net = slim.max_pool2d(net, ksize)
    return net

# somehow shape is inferred in the initializers.
def inference(x, keep_prob):
    with slim.arg_scope([slim.conv2d], padding = 'SAME', stride = 1, weights_initializer = tf.truncated_normal_initializer(stddev=0.1), biases_initializer= tf.zeros_initializer(), weights_regularizer=slim.l2_regularizer(0.0005)):
        with slim.arg_scope([slim.max_pool2d], padding = 'SAME', stride = 2): #kernel_size = 2, # I don't know whether you can set defaults for non-kwargs.
            x_image = tf.reshape(x, [-1,28,28,1])
            x1 = slim.stack(x_image, 
                            conv_and_pool, 
                            [(32, 5, 2), (64, 5, 2)],
                            scope="conv_and_pool")
            x1_flat = slim.flatten(x1) #keeps track of dimensions!
            x1_drop = slim.dropout(x1_flat, keep_prob = keep_prob)
            y = slim.fully_connected(x1_drop, 10, activation_fn=lambda x:x)
            return y
#one gripe: no composition of functions like f . g . h in python. (or >>=)

def loss(logits, labels):
  cross_entropy = tf.losses.softmax_cross_entropy(labels, logits) #note this is with logits, and one-hot labels
  #this already returns the mean
  return cross_entropy

def training(loss, learning_rate=1e-4):
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

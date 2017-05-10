# Use the default MNIST model which gets 99% accuracy

import keras as K
from keras import backend

import tensorflow as tf
#import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from tf_vars import *
from mnist_models import *

# ex. f = cnn_model
# assume that f already takes the softmax
# TODO: rewrite this without needing the batch size!
def mix(x, fs, batch_size=100): #128
    t = len(fs)
    ys = []
    for i in range(t):
        #with tf.variable_scope("mix_"+str(i)):
        #ys.append(tf.softmax(fs[i](x)))
        ys.append(fs[i](x))
    #this allows reuse
    # t
    #http://stackoverflow.com/questions/38222126/tensorflow-efficient-way-for-tensor-multiplication
    alphas = get_scope_variable('alphas', initializer=tf.constant_initializer(0.0),shape=[t])
                                #initializer=tf.truncated_normal_initializer(0,0.1), shape=[t])
                                #initializer=tf.constant(0.0,shape=[t]))
    #Am I doing the right optimization for the alphas?
    #alphas = tf.Variable(tf.constant(0, shape=t))
    ws = tf.nn.softmax(tf.exp(alphas))
    ws_reshaped = tf.reshape(ws, [1, t, 1])
    # batch_size * t * 1
    ws_tiled = tf.tile(ws_reshaped, [batch_size, 1, 1])
    # batch_size * labels * t
    ys1 = tf.stack(ys, axis=-1)
    # batch_size * labels 
    ys2 = tf.reshape(tf.matmul(ys1, ws_tiled), shape=[batch_size,-1])
    return (ys2, ws, ys1)

"""
def logt(x, t=0):
    return tf.log(tf.maximum(x, t))

#tf stopped providing this. Beware of instability?
def cross_entropy(y, yhat, t=0):
    #R^{l*n}, R^{l*n} -> R
    return tf.reduce_mean(-tf.reduce_sum(y * logt(yhat,t), reduction_indices=[-1]))
"""
#def entropy(y, t=0):
#    return tf.reduce_mean(-tf.reduce_sum(y * logt(y,t), reduction_indices=[-1]))
def entropy_reg(ws, t=0):
    return tf.reduce_mean(-logt(ws,t))
    #tf.nn.softmax(ws)
    #assume already did softmax

def mix_loss(y, ys1): # ws, reg_weight):
    return cross_entropy(y, ys1, 0.00001) # + reg_weight * entropy(ws)
#train using this loss!

def mix_reg(_, ws):
    return entropy_reg(ws, 0.00001)

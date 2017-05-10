import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

#from mnist_model import *

import os.path

import keras
import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.layers.pooling import *
from keras.layers.convolutional import *

from tf_utils import *
#from train_many import * 
from utils import *
from mnist_utils import *

def model1():
    nb_filters = 64
    input_shape = (28,28,1)
    nb_classes=10
    return Sequential([Dropout(0.2, input_shape=input_shape),
                       Conv2D(nb_filters, (8, 8), (2, 2), "same"),
                       Activation('relu'),
                       Conv2D((nb_filters * 2), (6, 6), (2, 2), "valid"),
                       Activation('relu'),
                       Conv2D((nb_filters * 2), (5, 5), (1, 1), "valid"),
                       Activation('relu'),
                       Dropout(0.5),
                       Flatten(),
                       Dense(nb_classes)])

#to deal with bug in keras https://github.com/fchollet/keras/issues/5268
def model1_no_dropout():
    return Sequential([
        Conv2D(64, (8, 8), input_shape=(28,28,1), strides = (2, 2), padding = "same"),
        Activation('relu'),
        Conv2D(128, (6, 6), strides = (2, 2), padding = "valid"),
        Activation('relu'),
        Conv2D(128, (5, 5), padding = "valid"),
        Activation('relu'),
        Flatten(),
        Dense(10),
        Activation('softmax')])

def model1_logits():
    return Sequential([
        Conv2D(64, (8, 8), input_shape=(28,28,1), strides = (2, 2), padding = "same"),
        Activation('relu'),
        Conv2D(128, (6, 6), strides = (2, 2), padding = "valid"),
        Activation('relu'),
        Conv2D(128, (5, 5), padding = "valid"),
        Activation('relu'),
        Flatten(),
        Dense(10)])

def model2():
    return Sequential([
        Conv2D(32, (5,5), input_shape=(28,28,1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Conv2D(64, (5,5),  padding='same', activation = 'relu'),
        MaxPooling2D(pool_size=(2, 2),  padding='same'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(10),
        Activation('softmax')])

# with L1 regularization
def model3(la = 0, ala = 0):
    return Sequential([
        Conv2D(64, (8, 8), input_shape=(28,28,1), strides = (2, 2), padding = "same",
               kernel_regularizer=regularizers.l1_l2(la),
               activity_regularizer=regularizers.l1(ala)),
        Activation('relu'),
        Conv2D(128, (6, 6), strides = (2, 2), padding = "valid",
               kernel_regularizer=regularizers.l1_l2(la),
               activity_regularizer=regularizers.l1(ala)),
        Activation('relu'),
        Conv2D(128, (5, 5), padding = "valid",
               kernel_regularizer=regularizers.l1_l2(la),
               activity_regularizer=regularizers.l1(ala)),
        Activation('relu'),
        Flatten(),
        Dense(10), 
               #kernel_regularizer=regularizers.l1_l2(la),
               #activity_regularizer=regularizers.l1(ala)),
        Activation('softmax')])

def model_sigmoid():
    return Sequential([
        Conv2D(64, (8, 8), input_shape=(28,28,1), strides = (2, 2), padding = "same"),
        Activation('sigmoid'),
        Conv2D(128, (6, 6), strides = (2, 2), padding = "valid"),
        Activation('sigmoid'),
        Conv2D(128, (5, 5), padding = "valid"),
        Activation('sigmoid'),
        Flatten(),
        Dense(10),
        Activation('softmax')])

def make_model_from_logits(model):
    def m(x,y):
        #print("Model:")
        #print(model)
        predictions = model(x)
        #labels are one-hot
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions), name = 'loss')
        acc = accuracy(y, predictions, vector=True)
        tf.add_to_collection('losses', loss)
        #added for comparing accuracies. This is batch_size * t
        #https://www.tensorflow.org/performance/xla/broadcasting
        #batch_size, t* batch_size. Must be this order to broadcast correctly
        #ind_correct = tf.cast(tf.transpose(tf.equal(tf.argmax(y,1), tf.transpose(tf.argmax(p_ind,1)))), tf.float32)
        #tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return {'loss': loss, 'inference': predictions, 'accuracy': acc}
        #, 'regularizer' : reg, 'ws': ws, 'ind_inference' : p_ind, 'ind_correct' : ind_correct}
    return m

def make_with_inputs(m):
    x, y = make_phs()
    d = m(x,y)
    ph_dict = {'x':x, 'y':y}
    return d, ph_dict

#DEPRECATED
def make_model(model):
    def m(x,y):
        #print("Model:")
        #print(model)
        predictions = model(x)
        loss = xe_loss(y, predictions)
        loss = tf.identity(loss, name="loss")
        acc = accuracy2(y, predictions)
        tf.add_to_collection('losses', loss)
        #added for comparing accuracies. This is batch_size * t
        #https://www.tensorflow.org/performance/xla/broadcasting
        #batch_size, t* batch_size. Must be this order to broadcast correctly
        #ind_correct = tf.cast(tf.transpose(tf.equal(tf.argmax(y,1), tf.transpose(tf.argmax(p_ind,1)))), tf.float32)
        #tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return {'loss': loss, 'inference': predictions, 'accuracy': acc}
        #, 'regularizer' : reg, 'ws': ws, 'ind_inference' : p_ind, 'ind_correct' : ind_correct}
    return m

#PROBLEM: I can't get y out of keras!
#OK if I don't vare about adversarial training
"""
def make_model_keras(model):
    def m(x,y):
        print("Model:")
        print(model)
        predictions = model(x)
        model.total_loss
        loss = xe_loss(y, predictions)
        loss = tf.identity(loss, name="loss")
        acc = accuracy2(y, predictions)
        tf.add_to_collection('losses', loss)
        #added for comparing accuracies. This is batch_size * t
        #https://www.tensorflow.org/performance/xla/broadcasting
        #batch_size, t* batch_size. Must be this order to broadcast correctly
        #ind_correct = tf.cast(tf.transpose(tf.equal(tf.argmax(y,1), tf.transpose(tf.argmax(p_ind,1)))), tf.float32)
        #tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return {'loss': loss, 'inference': predictions, 'accuracy': acc}
        #, 'regularizer' : reg, 'ws': ws, 'ind_inference' : p_ind, 'ind_correct' : ind_correct}
    return m
"""

#tf stopped providing this. Beware of instability?
def cross_entropy(y, yhat, t=0):
    """R^{l*n}, R^{l*n} -> R"""
    a = tf.reduce_mean(-tf.reduce_sum(y * logt(yhat,t), reduction_indices=[-1]))
    #a=tf.Print(a,[a,y,yhat],'a:', summarize=10)
    return a
        #tf.reduce_mean(-tf.reduce_sum(y * logt(yhat,t), reduction_indices=[-1]))

def xe_loss(y, ys1): # ws, reg_weight):
    return cross_entropy(y, ys1, 0.00001) # + reg_weight * entropy(ws)
#train using this loss!

def logt(x, t=0):
    return tf.log(tf.maximum(x, t))


"""
    layers = [Dropout(0.2, input_shape=input_shape),
              conv_2d(nb_filters, (8, 8), (2, 2), "same"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
              Activation('relu'),
              Dropout(0.5),
              Flatten(),
              Dense(nb_classes)]
"""

import tensorflow as tf

from cleverhans.utils_mnist import data_mnist

#import keras
from keras import backend as K

from utils import *
from tf_utils import *

def get_data_mnist(label_smooth=0, testing='F'):
    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    if testing == 'T':
        X_train = X_train[1:256]
        Y_train = Y_train[1:256]
    return X_train, Y_train, X_test, Y_test

def get_bf_mnist(label_smooth=0, testing='F'):
    X_train, Y_train, X_test, Y_test = get_data_mnist(label_smooth, testing)
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test})
    return train_data, test_data

def make_phs():
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    return x, y

def start_keras_session(learning_phase=0):
    K.set_learning_phase(learning_phase)
    sess = tf.Session()
    K.set_session(sess)
    return sess

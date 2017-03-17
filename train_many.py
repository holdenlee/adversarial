import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

from mnist_model import *

import os.path

import keras
from keras import backend

from utils import *

# X_train, Y_train, X_test, Y_test = data_mnist()
# train t neural nets whose model is f
# f, when called `f()`, should give a function, tensor -> tensor. Recommended: f is a Keras layer. Ex. f = cnn_model
def train_many(sess, f, t, X_train, Y_train, epochs = 1, X_test = None, Y_test = None, do_eval=False, train_params = {}, fn_model = None, cont = False, verbosity = 1):
    sess = tf.Session()
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    models = []
    if do_eval:
        accs = []
    for i in range(t):
        if cont and os.path.isfile(filepath + str(i) + '.hdf5'):
            continue
        printv('Training net '+str(i), verbosity, 1)
        model = f() #this is a Keras layer object, callable on tensors.
        predictions = model(x)
        #eval_fn = evaluate(sess, x, y, predictions, X_test, Y_test, batch_size = FLAGS.batch_size) if do_eval else None
        #I will actually decouple training from evaluation here
        model_train(sess, x, y, predictions, X_train, Y_train, #evaluate=eval_fn, 
                    args=train_params)
        if do_eval:
            #eval_params = {'batch_size': FLAGS.batch_size}
            #for simplicity, use same train_params as eval_params
            eval_params = train_params
            accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                                  args=eval_params)
            printv('Test accuracy on test examples: ' + str(accuracy),verbosity, 1)
            accs.append(accuracy)
        models.append(model)
        if fn_model != None:
            fn_model(model, i, accuracy)
    if do_eval:
        #print('do_eval, return models+acc')
        return (models, accs)
    else:
        #print('return models only')
        return models

def save_model_t(filepath, model, t, accuracy=None):
    model.save_weights(filepath + str(t) + '.hdf5')

def load_model_t(filepath, model, t):
    model.load_weights(filepath + str(t) + '.hdf5')

def train_many_and_save(filepath, sess, f, t, X_train, Y_train, epochs = 1, X_test = None, Y_test = None, do_eval=False, train_params = {}):
    output = train_many(sess, f, t, X_train, Y_train, epochs, X_test, Y_test, do_eval, train_params, fn_model = lambda model, t, accuracy: save_model_t(filepath, model, t, accuracy))
    #print(output)
    if do_eval:
        np.savetxt(filepath + '_accs.txt', output[1])
    return output


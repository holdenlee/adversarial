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

def train_many(f, t, X_train, Y_train, epochs = 1, X_test = None, Y_test = None, do_eval=False, fn_model = None, cont = False, verbosity = 1):
    models = []
    if do_eval:
        accs = []
    for i in range(t):
        if cont and os.path.isfile(filepath + str(i) + '.hdf5'):
            continue
        printv('Training net '+str(i), verbosity, 1)
        model = f() #this is a Keras layer object, callable on tensors.
        model.compile(optimizer='adadelta',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=epochs, batch_size=32)
        if do_eval:
            loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=128)
            printv(loss_and_metrics, verbosity, 1)
            accs.append(loss_and_metrics)
        models.append(model)
        if fn_model != None:
            fn_model(model, i, loss_and_metrics)
    if do_eval:
        return (models, accs)
    else:
        return models

def save_model_t(filepath, model, t, accuracy=None):
    model.save_weights(filepath + str(t) + '.hdf5')

def load_model_t(filepath, model, t):
    model.load_weights(filepath + str(t) + '.hdf5')

def train_many_and_save(filepath, f, t, X_train, Y_train, epochs = 1, X_test = None, Y_test = None, do_eval=False):
    output = train_many(f, t, X_train, Y_train, epochs, X_test, Y_test, do_eval, fn_model = lambda model, t, accuracy: save_model_t(filepath, model, t, accuracy))
    #print(output)
    if do_eval:
        np.savetxt(filepath + '_accs.txt', output[1])
    return output


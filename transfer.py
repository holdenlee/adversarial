import numpy as np

import tensorflow as tf

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

import os.path

import keras
from keras import backend

from train_many import * 
from utils import *
from mnist_utils import *

#model() should output logits
def make_adv_for_model(X_test, Y_test, filename, model, ep = 0.3, adv_f = fgsm2_clip, keras=True):
    if keras:
        sess = start_keras_session()
    else:
        sess = tf.Session()
    #must do all this before loading model
    x, y = make_phs()
    m = model()
    d, epsilon = make_adversarial_model(make_model_from_logits(m), fgsm2_clip, x, y)
    #does it matter whether we do this before or afterwards?
    if keras:
        m.load_weights(filename)
    else: 
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(filename))
    #    model.load_weights(filepath, by_name=True)
    print('Loaded model from %s' % filename)
    adv_exs = sess.run(d['adv_x'], {x:X_test, y:Y_test, epsilon: ep})
    return adv_exs

#model() is a function taking in x, y.
#model2() is a function that creates x, y.
def mnist_adv_transfer(filepath, model, model2, ep = 0.3, adv_f = fgsm2_clip, keras=True, folder='./', label_smooth=0, keras2=False):
    X_train, Y_train, X_test, Y_test = get_data_mnist(label_smooth, testing='F')
    adv_exs = make_adv_for_model(X_test, Y_test, filepath, model, ep, adv_f, keras)
    tf.reset_default_graph()
    acc = eval_model_on(tf.train.latest_checkpoint(folder), model2, adv_exs, Y_test, keras2)

def eval_model_on(filepath, model_f, adv_exs, Y_test, keras2=False):
    if keras2:
        sess = start_keras_session(0)
    else:
        sess = tf.Session()
    #2 Restoring MWU
    d2, ph_dict = model_f()
    #, epvar = mnist_mwu_model_adv(fgsm, lambda: mnist_mwu_model(u=20,u2=20)) #the adversarial part doesn't matter
    saver = tf.train.Saver()
    #with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, filepath)
    print("Model restored.")
    #acc = transfer(sess, d, d2, epsilon, ph_dict['x'], ph_dict['y'], X_test, Y_test, 0)
    acc = sess.run(d2['accuracy'], {ph_dict['x']:adv_exs, ph_dict['y']:Y_test})
    print("Accuracy: %f" % acc)
    return acc

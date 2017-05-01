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

from train_many import * 
from utils import *

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', 'pretrain5_model1_smooth0/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'nets', 'Filename to save model under.')
flags.DEFINE_string('save_as', 'experiment_adv_transfer_mwu/transfer.txt', 'Save as')
flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_float('ep', 0.3, 'Learning rate for training')
flags.DEFINE_integer('t', 2, 'Number of nets')
flags.DEFINE_string('testing', 'F', 'Run as test on small training set to make sure code works')
flags.DEFINE_integer('label_smooth', 0.1, 'How much to clip y values (0 for no clipping)')

if __name__=='__main__':
    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    label_smooth = FLAGS.label_smooth
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    if FLAGS.testing == 'T':
        X_train = X_train[1:256]
        Y_train = Y_train[1:256]
    # Define input TF placeholder
    #x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    #y = tf.placeholder(tf.float32, shape=(None, 10))
    #1 Restoring conv
    filepath = FLAGS.train_dir+FLAGS.filename
    K.set_learning_phase(0)
    sess = tf.Session()
    K.set_session(sess)
    #must do all this before loading model
    model = model1_no_dropout  
    #cnn_model #BUG with keras
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    m = model()
    #load weights?
    #model.load_weights(filepath + str(t) + '.hdf5')
    load_model_t(filepath, m, 0)
    #    model.load_weights(filepath, by_name=True)
    print('Loaded model %d' % 0)
    d, epsilon = make_adversarial_model(make_model(m), adv, x, y)
    #2 Restoring MWU
    saver = tf.train.Saver()

    d2, ph_dict, epvar = mnist_mwu_model_adv(lambda: mnist_mwu_model(u=20,u2=20), fgsm) #the adversarial part doesn't matter
    with tf.Session() as sess:
    # Restore variables from disk.
        saver.restore(sess, "train_adv_mwu/model.ckpt")
        print("Model restored.")
    acc = transfer(sess, d, d2, epsilon, ph_dict['x'], ph_dict['y'], X_test, Y_test, 0)
    print("Accuracy: %f" % acc)

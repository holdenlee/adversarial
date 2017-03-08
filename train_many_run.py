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

flags.DEFINE_string('train_dir', 'pretrain/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'nets', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_integer('t', 2, 'Number of nets')
flags.DEFINE_string('testing', 'F', 'Run as test on small training set to make sure code works')
flags.DEFINE_integer('label_smooth', 0.1, 'How much to clip y values (0 for no clipping)')

# ^ Do I need to include these in every file, or just in main file?

default_train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }

#if __name__ == '__main__': 
def main(argv=None):
    sess = tf.Session()
    keras.backend.set_session(sess)

    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    label_smooth = FLAGS.label_smooth
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    if FLAGS.testing == 'T':
        X_train = X_train[1:256]
        Y_train = Y_train[1:256]
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    filepath = FLAGS.train_dir+FLAGS.filename
    models, accs = train_many_and_save(filepath, sess, cnn_model, FLAGS.t, X_train, Y_train, epochs = 1, X_test = X_test, Y_test = Y_test, do_eval=True, train_params = default_train_params)

if __name__ == '__main__':
    app.run()

"""
# Run test
python train_many.py --testing=T
# Run for real
python train_many.py --t=100 2>&1 | tee pretrain/log.txt
# Nohup
nohup python train_many.py --t=100 > pretrain/log.txt 2>&1 &
#
program [arguments...] 2>&1 | tee outfile
"""

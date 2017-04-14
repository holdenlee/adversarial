import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

from mnist_model import *
from mnist_models import *

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

#if __name__ == '__main__': 
def main(argv=None):
    #sess = tf.Session()
    #keras.backend.set_session(sess)
    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    label_smooth = FLAGS.label_smooth
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    if FLAGS.testing == 'T':
        X_train = X_train[1:256]
        Y_train = Y_train[1:256]
    filepath = FLAGS.train_dir+FLAGS.filename
    fors_([[0,0.00001, 0.00003, 0.0001, 0.0003, 0.001],
           [0,0.00001, 0.00003, 0.0001, 0.0003, 0.001]],
           lambda li: train_one(lambda: model3(li[0], li[1]), 
                                str(li[0])+"_"+str(li[1]), X_train, Y_train, epochs = FLAGS.nb_epochs, X_test = X_test, Y_test = Y_test, do_eval=True, fn_model = (lambda model, t, accuracy: save_model_t(filepath, model, t))))
#def train_many_and_save(filepath, f, t, X_train, Y_train, epochs = 1, X_test = None, Y_test = None, do_eval=False, train_params = {}):
"""
[0,0.001, 0.01, 0.03, 0.1, 0.3],
           [0,0.001, 0.01, 0.03, 0.1, 0.3]
"""

if __name__ == '__main__':
    app.run()

"""
train_one(f, s, X_train, Y_train, epochs = 1, X_test = None, Y_test = None, do_eval=False, fn_model = None, cont = False, verbosity = 1)
"""

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

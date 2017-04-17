import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
#from tensorflow.examples.tutorials.mnist import input_data

from cleverhans.utils_mnist import data_mnist
#from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

import os.path

from mnist_models import *
from mix import *
from adv_model import *
from tf_utils import *
from train_many import *
from random import random
from train_single import *

import keras.backend as K

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', 'train_single/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'single.ckpt', 'Filename to save model under.')
#flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 100, 'Size of training batches. Must divide evenly into the dataset sizes. (FIX this later)')
#for batch_size = 100, is 600 times nb_epochs
#STEPS
flags.DEFINE_integer('max_steps', 3600, 'Number of steps to run trainer.')
flags.DEFINE_integer('print_steps', 100, 'Print progress every...')
flags.DEFINE_integer('eval_steps', 600, 'Run evaluation every...')
flags.DEFINE_integer('save_steps', 1200, 'Run evaluation every...')
flags.DEFINE_integer('summary_steps', 1200, 'Run summary every...')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_integer('verbosity', 1, 'How chatty')
flags.DEFINE_float('label_smooth', 0.1, 'How much to clip y values (0 for no clipping)')
flags.DEFINE_float('epsilon', 0.3, 'Strength of attack')
flags.DEFINE_string('clip', 'T', 'Whether to clip values to [0,1]')
flags.DEFINE_string('fake_data', 'F', 'Use fake data.  ')
flags.DEFINE_string('testing', 'F', 'Test mode.  ')


def main(_):
    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    label_smooth = FLAGS.label_smooth
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    if FLAGS.testing == 'T':
        X_train = X_train[1:256]
        Y_train = Y_train[1:256]
    #K.set_learning_phase(1)
    #sess = tf.Session()
    #K.set_session(sess)
    #?? There is a problem with fgsm_clip
    train_single(model_sigmoid, fgsm2_clip, X_train, Y_train, X_test, Y_test, label_smooth = FLAGS.label_smooth, batch_size = FLAGS.batch_size, eval_steps = FLAGS.eval_steps, learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon, print_steps=FLAGS.print_steps, save_steps =FLAGS.save_steps, train_dir = FLAGS.train_dir, filename = FLAGS.filename, summary_steps=FLAGS.summary_steps, max_steps=FLAGS.max_steps, verbosity=FLAGS.verbosity)
    #train_one(cnn_model, '', X_train, Y_train, epochs = FLAGS.nb_epochs, X_test = X_test, Y_test = Y_test, do_eval=True, fn_model = None, cont = False, verbosity = 1)

if __name__=='__main__':
    app.run()

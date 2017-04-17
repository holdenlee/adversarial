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
from train_single import *

FLAGS = flags.FLAGS

flags.DEFINE_string('file_in','pretrain5_model1_smooth0/nets0', 'Filename where model is saved.')
flags.DEFINE_string('train_dir', 'experiment_train_on_adv/', 'Directory to save model.')
flags.DEFINE_string('filename', 'model.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
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
flags.DEFINE_float('label_smooth', 0, 'How much to clip y values (0 for no clipping)')
flags.DEFINE_float('epsilon', 0.3, 'Strength of attack')
#flags.DEFINE_string('clip', 'T', 'Whether to clip values to [0,1]')
flags.DEFINE_string('fake_data', 'F', 'Use fake data.  ')
flags.DEFINE_string('testing', 'F', 'Use fake data.  ')

def main(_):
    #does clipping.
    sess, m, d, epsilon, x, y = load_one(model1_no_dropout, FLAGS.file_in, adv=fgsm2_clip)
    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    label_smooth = FLAGS.label_smooth
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    if FLAGS.testing == 'T':
        X_train = X_train[1:256]
        Y_train = Y_train[1:256]
    X_advs=[]
    for i in range(len(X_train)):
        X_adv = sess.run((d['adv_x']), feed_dict = {x:X_train[i:i+1], y:Y_train[i:i+1], epsilon:FLAGS.epsilon}) #.flatten()
        X_advs.append(X_adv)        
    #X_adv = sess.run(d['adv_x'], {x:X_train, y:Y_train, epsilon: FLAGS.epsilon})
    #otherwise gives feed error
    #tf.reset_default_graph()
    X_advs = np.concatenate(X_advs)
    K.set_learning_phase(1)
    sess = tf.Session()
    K.set_session(sess)
    X2 = np.concatenate([X_train,X_advs])
    Y2 = np.concatenate([Y_train,Y_train])
    train_one(cnn_model, '', X2, Y2, epochs = FLAGS.nb_epochs, X_test = X_test, Y_test = Y_test, do_eval=True, fn_model = None, cont = False, verbosity = 1)
    #train_single(X_adv, Y_train, X_test, Y_test, label_smooth = FLAGS.label_smooth, batch_size = FLAGS.batch_size, eval_steps = FLAGS.eval_steps, learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon, print_steps=FLAGS.print_steps, save_steps =FLAGS.save_steps, train_dir = FLAGS.train_dir, filename = FLAGS.filename, summary_steps=FLAGS.summary_steps, max_steps=FLAGS.max_steps, verbosity=FLAGS.verbosity)

if __name__=='__main__':
    app.run()

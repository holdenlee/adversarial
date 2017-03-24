import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
#from tensorflow.examples.tutorials.mnist import input_data

from cleverhans.utils_mnist import data_mnist
#from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

#from mnist_model import *
from mix import *
from adv_model import *
from tf_utils import *
from train_many import *
from random import random
from train_mix import *

import keras.backend as K

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', 'train_mix2/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mix.ckpt', 'Filename to save model under.')
flags.DEFINE_string('adv', 'fgsm', 'Type of adversary.')
#flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 100, 'Size of training batches. Must divide evenly into the dataset sizes. (FIX this later)')
#for batch_size = 100, is 600 times nb_epochs
flags.DEFINE_integer('max_steps', 3600, 'Number of steps to run trainer.')
flags.DEFINE_integer('print_steps', 100, 'Print progress every...')
flags.DEFINE_integer('eval_steps', 600, 'Run evaluation every...')
flags.DEFINE_integer('save_steps', 1200, 'Run evaluation every...')
flags.DEFINE_integer('summary_steps', 1200, 'Run summary every...')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_integer('verbosity', 1, 'How chatty')
flags.DEFINE_string('load', 'T', 'Whether to load nets')
flags.DEFINE_string('load_from', 'pretrain/nets', 'Whether to run control experiment')
flags.DEFINE_integer('t', 100, 'Number of nets')
flags.DEFINE_integer('label_smooth', 0.1, 'How much to clip y values (0 for no clipping)')
flags.DEFINE_float('reg_weight', 1, 'Weight on entropy regularizer')
flags.DEFINE_float('epsilon', 0.3, 'Strength of attack')
tf.app.flags.DEFINE_string('fake_data', False, 'Use fake data.  ')

def main(_):
    #print(FLAGS)
    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    Y_train = Y_train.clip(FLAGS.label_smooth / 9., 1. - FLAGS.label_smooth)
    trainer = make_trainer(X_train, Y_train, X_test, Y_test, adv=FLAGS.adv, t=FLAGS.t, load=FLAGS.load, load_from=FLAGS.load_from, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate, save_steps = FLAGS.save_steps, summary_steps=FLAGS.summary_steps, train_dir='train/', eval_steps=FLAGS.eval_steps, print_steps = FLAGS.print_steps, epsilon=FLAGS.epsilon, max_steps = FLAGS.max_steps, reg_weight = FLAGS.reg_weight, verbosity=FLAGS.verbosity)
    trainer.init_and_train()
    trainer.finish()

if __name__=='__main__':
    app.run()

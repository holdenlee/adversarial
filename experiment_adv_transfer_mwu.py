import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

#from mnist_model import *

import os.path

import keras
from keras import backend

from train_many import * 
from utils import *

from mwu import *
from mwu_adv import *

from transfer import *

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', 'pretrain5_model1_smooth0/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'nets0.hdf5', 'Filename to save model under.')
flags.DEFINE_string('save_as', 'experiment_adv_transfer_mwu/transfer.txt', 'Save as')
flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_float('ep', 0.3, 'Learning rate for training')
flags.DEFINE_integer('t', 2, 'Number of nets')
flags.DEFINE_string('testing', 'F', 'Run as test on small training set to make sure code works')
flags.DEFINE_integer('label_smooth', 0.1, 'How much to clip y values (0 for no clipping)')
flags.DEFINE_string('to_file', "./train_adv_mwu/", 'transfer to model')

if __name__=='__main__':
    filepath = os.path.join(FLAGS.train_dir,FLAGS.filename)
    model = model1_logits
    def model2():
        d2, ph_dict, epvar = mnist_mwu_model_adv(fgsm, lambda: mnist_mwu_model(u=20,u2=20))
        return d2, ph_dict
    mnist_adv_transfer(filepath, model, model2, ep = 0.3, adv_f = fgsm2_clip, keras=True, folder=FLAGS.to_file, label_smooth=FLAGS.label_smooth)

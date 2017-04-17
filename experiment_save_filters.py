import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

#from mnist_model import *
from mnist_models import *

import os.path

import keras
from keras import backend

from train_many import *
from mix import *
from train_mix import *
from utils import *
from PIL import Image
from mnist_tests import *

if __name__=='__main__':
    np.set_printoptions(threshold=np.nan, precision=5)
    #cnn_model_summary()
    #* Examine filters
    #examine_filters(cnn_model())
    X_train, Y_train, X_test, Y_test = data_mnist()
    sess, models, dicts, eps, x, y = load_many(model1_no_dropout, filepath='pretrain5_model1_smooth0/nets', adv = fgsm2_clip, t=1)
    save_filters(models[0], 'experiment_save_filters/filter')
    #for (i,m) in enumerate(models):
    #    print("Model %d" %i)
    #    examine_filters(m)


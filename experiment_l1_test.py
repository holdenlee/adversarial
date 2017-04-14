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

flags.DEFINE_string('train_dir', 'pretrain5_model1_smooth0/nets', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'nets', 'Filename to save model under.')
flags.DEFINE_string('save_as', 'experiment_adv_transfer/transfer.txt', 'Save as')
flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_float('ep', 0.3, 'Epsilon')
flags.DEFINE_string('testing', 'F', 'Run as test on small training set to make sure code works')
flags.DEFINE_integer('label_smooth', 0.1, 'How much to clip y values (0 for no clipping)')

def l1_test(X, Y, filepath='experiment_l1/nets', save_as = 'experiment_l1_test/accs.txt', adv = fgsm2_clip, ep = 0.3):
    K.set_learning_phase(0)
    sess = tf.Session()
    K.set_session(sess)
    #must do all this before loading model
    model = model1_no_dropout  
    #cnn_model #BUG with keras
    models = []
    dicts = []
    eps = []
    params = []
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    for la in [0,0.00001, 0.00003, 0.0001, 0.0003, 0.001]:
        for ala in [0,0.00001, 0.00003, 0.0001, 0.0003, 0.001]:
            m = model()
            models.append(m)
            #load weights?
            load_model_t(filepath, m, str(la)+'_'+str(ala))
            #    model.load_weights(filepath, by_name=True)
            print('Loaded model %s' % (str(la)+'_'+str(ala)))
            d, epsilon = make_adversarial_model(make_model(m), adv, x, y)
            dicts.append(d)
            eps.append(epsilon)
            params.append((la,ala))
    accs = []
    for i in range(len(models)):
        d = dicts[i]
        epsilon = eps[i]
        (la, ala) = params[i]
        acc = sess.run(d['adv_accuracy'], {x:X_test, y:Y_test, epsilon: ep})
        print("Accuracy for %s: %f" % ((str(la)+'_'+str(ala)), acc))
        accs.append(acc)
    np.savetxt(save_as, accs)

if __name__=='__main__':
    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    #label_smooth = FLAGS.label_smooth
    #Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    #if FLAGS.testing == 'T':
    #    X_train = X_train[1:256]
    #    Y_train = Y_train[1:256]
    filepath = FLAGS.train_dir+FLAGS.filename
    l1_test(X_test, Y_test, filepath=filepath, save_as=FLAGS.save_as, adv = fgsm2_clip, ep = FLAGS.ep)

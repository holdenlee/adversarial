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
flags.DEFINE_float('ep', 0.3, 'Learning rate for training')
flags.DEFINE_integer('t', 2, 'Number of nets')
flags.DEFINE_string('testing', 'F', 'Run as test on small training set to make sure code works')
flags.DEFINE_integer('label_smooth', 0.1, 'How much to clip y values (0 for no clipping)')


def transfer(sess, d1, d2, ep1, x, y, X_test, Y_test, epsilon):
    #adv_xs=[]
    #for i in range(t):
    #    adv_x = sess.run((d['adv_x']), feed_dict = {x:X[i:i+1], y:Y[i:i+1]}).flatten()
    #    adv_xs.append(adv_x)
    adv_exs = sess.run(d1['adv_x'], {x:X_test, y:Y_test, ep1: epsilon})
    acc = sess.run(d2['accuracy'], {x:adv_exs, y:Y_test})
    return acc    

"""
    K.set_learning_phase(0)
    sess = tf.Session()
    K.set_session(sess)
    #must do all this before loading model
    model = model1_no_dropout() #BUG
    #model.summary()
    model.load_weights(filepath, by_name=True)
    #print(model.get_weights())
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    #def m(x,y):
    #    loss = mix_loss(y, predictions)
    #    return {'loss': loss, 'inference': predictions, 'accuracy': acc, 'regularizer' : reg, 'ws': ws, 'ind_inference' : p_ind, 'ind_correct' : ind_correct}
    d, epsilon = make_adversarial_model(make_model(model), adv, x, y)

"""


def transfers(X, Y, filepath='pretrain5_model1_smooth0/nets', save_as = 'experiment_adv_transfer/transfer.txt', adv = fgsm2_clip, ep = 0.3, t=100):
    K.set_learning_phase(0)
    sess = tf.Session()
    K.set_session(sess)
    #must do all this before loading model
    model = model1_no_dropout  
    #cnn_model #BUG with keras
    models = []
    dicts = []
    eps = []
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    for i in range(t):
        models.append(model())
        #load weights?
        load_model_t(filepath, models[i], i)
        #    model.load_weights(filepath, by_name=True)
        print('Loaded model %d' % i)
        d, epsilon = make_adversarial_model(make_model(models[i]), adv, x, y)
        dicts.append(d)
        eps.append(epsilon)
    d1 = dicts[0]
    ep1 = eps[0]
    accs = []
    for i in range(t):
        d2 = dicts[i]
        acc = transfer(sess, d1, d2, ep1, x, y, X_test, Y_test, ep)
        print("Accuracy for %d: %f" % (i, acc))
        accs.append(acc)
    np.savetxt(save_as, accs)

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
    filepath = FLAGS.train_dir+FLAGS.filename
    transfers(X_test, Y_test, filepath=filepath, save_as=FLAGS.save_as, adv = fgsm2_clip, ep = FLAGS.ep, t=FLAGS.t)

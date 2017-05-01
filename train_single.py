import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
#from tensorflow.examples.tutorials.mnist import input_data

from cleverhans.utils_mnist import data_mnist
#from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

import os.path

#from mnist_model import *
from mix import *
from adv_model import *
from tf_utils import *
from train_many import *
from random import random

import keras.backend as K

def fgsm_clip(x, predictions, eps):
    return fgsm(x,predictions,eps,0,1)

def make_batch_feeder_ep(args, ep_f, refresh_f=shuffle_refresh, num_examples = None):
    if num_examples==None:
        l = len(list(args.values())[0]) #fix for python 3
    else:
        l = num_examples
    def f(bf, batch_size): 
        d = batch_feeder_f(bf, batch_size, refresh_f)
        d['epsilon'] = ep_f()
    return BatchFeeder(args, l, f)

def single_model(adv_f = fgsm, f=cnn_model):
    m = f()
    #THIS MUST BE OUTSIDE otherwise 2 copies will be made!
    def model(x,y):
        print("Model")
        print(m)
        print(x)
        print(y)
        predictions = m(x)
        loss = cross_entropy(y, predictions, 0.00001)
        loss = tf.identity(loss, name="loss")
        acc = accuracy2(y, predictions)
        tf.add_to_collection('losses', loss)
        return {'loss': loss, 'inference': predictions, 'accuracy': acc}
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    adv_model, epsilon = make_adversarial_model(model, adv_f, x, y)
    ph_dict = {'x': x, 'y': y, 'epsilon': epsilon}
    return adv_model, ph_dict, epsilon

def train_single(f, adv_f, X_train, Y_train, X_test, Y_test, label_smooth = 0, batch_size = 100, eval_steps = 600, learning_rate=0.1, epsilon=0.3, print_steps=100, save_steps =1200, train_dir = 'train_single/', filename = 'model.ckpt', summary_steps=1200, max_steps=3600, verbosity=1):
    #X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    #train_data = make_batch_feeder({'x': X_train, 'y':Y_train}, lambda: 0.5 * random())
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test}) #don't need to shuffle here
    # https://github.com/fchollet/keras/issues/2310
    K.set_learning_phase(1)
    sess = tf.Session()
    keras.backend.set_session(sess)
    adv_model, ph_dict, epvar = single_model(adv_f, f)
    evals = [Eval(test_data, batch_size, ['adv_accuracy'], eval_feed={'epsilon': i*0.1}, eval_steps = eval_steps, name="test (adversarial %f)" % (i*0.1)) for i in range(1,6)]
    addons = [GlobalStep(),
                TrackAverages(), #do this before train (why?)
                Train(lambda gs: tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1e-08), batch_size, train_feed={'epsilon' : epsilon}, loss = 'combined_loss', print_steps=print_steps),
                Histograms(), #includes gradients, so has to be done after train
                Saver(save_steps = save_steps, checkpoint_path = os.path.join(train_dir, filename)),
                SummaryWriter(summary_steps = summary_steps, feed_dict = {}), #'keep_prob': 1.0
                Eval(test_data, batch_size, ['accuracy'], eval_feed={}, eval_steps = eval_steps, name="test (real)")] + evals
    trainer = Trainer(adv_model, max_steps, train_data, addons, ph_dict, train_dir = train_dir, verbosity=verbosity, sess=sess)
    trainer.init_and_train()
    trainer.finish()

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
from plots import *

import keras.backend as K

FLAGS = flags.FLAGS

flags.DEFINE_integer('label_smooth', 0.1, 'How much to clip y values (0 for no clipping)')

def f1(X_train, Y_train, X_test, Y_test, t, ep, reg, lr):
    tf.reset_default_graph()
    print("t: %d, epsilon: %.1f, reg: %.1f, lr: %.1f" % (t, ep, reg, lr))
    name = 'mix%d_pretrain1_ep%.1f_reg%.1f_lr%.1f' % (t, ep, reg, lr)
    trainer = make_trainer(X_train, Y_train, X_test, Y_test, adv='fgsm', t=t, learning_rate=lr, train_dir='mix/1/'+name, epsilon=ep, max_steps = 6000, reg_weight=reg)
    trainer.init_and_train()
    trainer.finish()
    return trainer

def f1_names(t,ep, reg,lr):
    return ("Mix %d, pretrained, fgsm %.1f, reg %.1f, lr %.1f" % (t, ep, reg, lr), 'mix/1/mix%d_pretrain1_ep%.1f_reg%.1f_lr%.1f' % (t, ep, reg, lr))

if __name__=='__main__':
    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    Y_train = Y_train.clip(FLAGS.label_smooth / 9., 1. - FLAGS.label_smooth)
    params = [[5, 10, 20], # t
              [0.2, 0.5, 1, 2], # reg
              [0.1, 0.2, 0.5, 1]] # lr
    fors_(params, 
          lambda li: f1(X_train, Y_train, X_test, Y_test, li[0], 0.3, li[1], li[2]))
    names = fors(params, lambda li: f1_names(li[0], 0.3, li[1], li[2]))
    print("names: ", names)
    make_plots(names, 'plots/1/')
    #print("end")

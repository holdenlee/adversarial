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
from train_single import *

import keras.backend as K

from mwu import *
from mwu_adv import *

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
flags.DEFINE_string('fake_data', False, 'Use fake data.  ')

def train_mwu(f, adv_f, X_train, Y_train, X_test, Y_test, label_smooth = 0, batch_size = 100, eval_steps = 600, learning_rate=0.1, epsilon=0.3, print_steps=100, save_steps =1200, train_dir = 'train_single/', filename = 'model.ckpt', summary_steps=1200, max_steps=3600, verbosity=1):
    #X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    #train_data = make_batch_feeder({'x': X_train, 'y':Y_train}, lambda: 0.5 * random())
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test}) #don't need to shuffle here
    # https://github.com/fchollet/keras/issues/2310
    #K.set_learning_phase(1)
    sess = tf.Session()
    #keras.backend.set_session(sess)
    adv_model, ph_dict, epvar = mnist_mwu_model_adv(adv_f, f)
    evals = [Eval(test_data, batch_size, ['adv_accuracy'], eval_feed={'epsilon': i*0.1}, eval_steps = eval_steps, name="test (adversarial %f)" % (i*0.1)) for i in range(1,6)]
    Ws = tf.get_collection('Ws')
    bs = tf.get_collection('bs')
    #print("collections:"+str((Ws,bs)))
    opter = lambda: MWOptimizer(Ws, bs, learning_rate=learning_rate, smoothing=0)
    addons = [Train(opter, 
                    batch_size, 
                    train_feed={'epsilon' : FLAGS.epsilon},
                    loss = 'combined_loss', 
                    print_steps=print_steps),
              Saver(save_steps = save_steps, checkpoint_path = os.path.join(train_dir, filename)),
                #SummaryWriter(summary_steps = summary_steps, feed_dict = {}), #'keep_prob': 1.0
              Eval(test_data, batch_size, ['accuracy'], eval_feed={}, eval_steps = eval_steps, name="test (real)")] + evals
    trainer = Trainer(adv_model, max_steps, train_data, addons, ph_dict, train_dir = train_dir, verbosity=verbosity, sess=sess)
    trainer.init_and_train()
    trainer.finish()

def main(_):
    X_train, Y_train, X_test, Y_test = data_mnist()
    train_mwu(lambda: mnist_mwu_model(u=50,u2=20), fgsm, X_train, Y_train, X_test, Y_test, label_smooth = 0, batch_size = 100, eval_steps = 600, learning_rate=0.1, epsilon=0.03, print_steps=100, save_steps =1200, train_dir = 'train_adv_mwu_50/', filename = 'model.ckpt', summary_steps=1200, max_steps=3600, verbosity=1)

if __name__=='__main__':
    app.run()

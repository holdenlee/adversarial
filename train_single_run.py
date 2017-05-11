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
from mnist_utils import *

import keras.backend as K

FLAGS = flags.FLAGS

#NOTE: THIS NEEDS ./
flags.DEFINE_string('train_dir', './train_single/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'model.ckpt', 'Filename to save model under.')
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
flags.DEFINE_float('label_smooth', 0, 'How much to clip y values (0 for no clipping)')
flags.DEFINE_float('epsilon', 0.3, 'Strength of attack')
flags.DEFINE_string('clip', 'T', 'Whether to clip values to [0,1]')
flags.DEFINE_string('fake_data', False, 'Use fake data.  ')
flags.DEFINE_string('testing', 'F', 'Use fake data.  ')

def main(_):
    train_data, test_data = get_bf_mnist(FLAGS.label_smooth, FLAGS.testing)
    sess = start_keras_session(1)
    model = model1_logits()
    adv_model, ph_dict, epsilon = make_adversarial_model_from_logits_with_inputs(model, fgsm2_clip)
    #adv_model, ph_dict, epsilon =  make_adversarial_model(make_model_from_logits(m), fgsm2_clip, x, y)
    #single_model()
    #print(adv_model)
    evals = [Eval(test_data, FLAGS.batch_size, ['adv_accuracy'], eval_feed={'epsilon': i*0.1}, eval_steps = FLAGS.eval_steps, name="test (adversarial %f)" % (i*0.1)) for i in range(1,6)]
    addons = [GlobalStep(),
                #TrackAverages(), #do this before train (why?)
                Train(lambda gs: tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate, rho=0.95, epsilon=1e-08), FLAGS.batch_size, train_feed={'epsilon' : FLAGS.epsilon}, loss = 'combined_loss', print_steps=FLAGS.print_steps),
                Histograms(), #includes gradients, so has to be done after train
                Saver(save_steps = FLAGS.save_steps, checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.filename)),
#os.path.join(FLAGS.train_dir, 'model.ckpt')),
                SummaryWriter(summary_steps = FLAGS.summary_steps, feed_dict = {}), #'keep_prob': 1.0
                Eval(test_data, FLAGS.batch_size, ['accuracy'], eval_feed={}, eval_steps = FLAGS.eval_steps, name="test (real)")] + evals
    #           Eval(test_data, FLAGS.batch_size, ['adv_accuracy'], eval_feed={'epsilon': FLAGS.epsilon}, eval_steps = FLAGS.eval_steps, name="test (adversarial)")]
                # + evals
    #pl_dict, model = adv_mnist_fs()
    #print(FLAGS)
    #print(FLAGS.epsilon)
    #print(FLAGS.train_dir)
    trainer = Trainer(adv_model, FLAGS.max_steps, train_data, addons, ph_dict, train_dir = FLAGS.train_dir, verbosity=FLAGS.verbosity, sess=sess)
    trainer.init_and_train()
    trainer.finish()

if __name__=='__main__':
    app.run()

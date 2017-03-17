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

import keras.backend as K

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', 'train_mix2/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mix.ckpt', 'Filename to save model under.')
#flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 100, 'Size of training batches. Must divide evenly into the dataset sizes. (FIX this later)')
#for batch_size = 100, is 600 times nb_epochs
flags.DEFINE_integer('max_steps', 3600, 'Number of steps to run trainer.')
flags.DEFINE_integer('eval_steps', 600, 'Run evaluation every...')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_string('load', 'T', 'Whether to load nets')
flags.DEFINE_string('load_from', 'pretrain/nets', 'Whether to run control experiment')
flags.DEFINE_integer('t', 100, 'Number of nets')
flags.DEFINE_integer('label_smooth', 0.1, 'How much to clip y values (0 for no clipping)')
flags.DEFINE_float('reg_weight', 1, 'Weight on entropy regularizer')
flags.DEFINE_float('epsilon', 0.3, 'Strength of attack')
tf.app.flags.DEFINE_string('fake_data', False, 'Use fake data.  ')


class Logger(AddOn):
    def __init__(self, log_steps=100):
        self.log_steps=log_steps
    def init(self, trainer):
        trainer.gets_dict['ws'] = trainer.data['ws']
        #trainer.gets_dict['inference'] = trainer.data['inference']
        #trainer.gets_dict['ind_inference'] = trainer.data['ind_inference']
        return True
    def run(self, trainer):
        step = trainer.step
        if (valid_pos_int(self.log_steps) and step % self.log_steps == 0) or (step + 1) == trainer.max_step:
            #printv("Predictions: %s" % str(trainer.outputs['inference']), trainer.verbosity, 1)
            #printv("Predictions: %s" % str(trainer.outputs['ind_inference']), trainer.verbosity, 1)
            printv("Weights: %s" % str(trainer.outputs['ws']), trainer.verbosity, 1)
        return True

def make_batch_feeder_ep(args, ep_f, refresh_f=shuffle_refresh, num_examples = None):
    if num_examples==None:
        l = len(list(args.values())[0]) #fix for python 3
    else:
        l = num_examples
    def f(bf, batch_size): 
        d = batch_feeder_f(bf, batch_size, refresh_f)
        d['epsilon'] = ep_f()
    return BatchFeeder(args, l, f)

evals = [Eval(test_data, FLAGS.batch_size, ['adv_accuracy'], eval_feed={'epsilon': i*0.1}, eval_steps = 1000, name="test (adversarial @ %f)" % (i*0.1)) for i in range(1,6)]

def main(_):
    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    Y_train = Y_train.clip(FLAGS.label_smooth / 9., 1. - FLAGS.label_smooth)
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    #train_data = make_batch_feeder({'x': X_train, 'y':Y_train}, lambda: 0.5 * random())
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test}) #don't need to shuffle here
    # https://github.com/fchollet/keras/issues/2310
    #K._LEARNING_PHASE = tf.constant(0)
    K.set_learning_phase(1)
    sess = tf.Session()
    keras.backend.set_session(sess)
    model = cnn_model
    #NO!!!
    #models = FLAGS.t*[model()]
    models = []
    for i in range(FLAGS.t):
        models.append(model())
    #Do I need to start the session first?
    if FLAGS.load == 'T':
        for i in range(FLAGS.t):
            load_model_t(FLAGS.load_from, models[i], i)
    def model(x, y):
        predictions, ws, p_ind = mix(x, models, FLAGS.batch_size)
        loss = mix_loss(y, predictions)
        loss = tf.identity(loss, name="loss")
        reg = FLAGS.reg_weight * entropy_reg(ws, 0.00001)
        reg = tf.identity(reg, name="regularizer")
        acc = accuracy2(y, predictions)
        tf.add_to_collection('losses', loss)
        tf.add_to_collection('losses', reg)
        return {'loss': loss, 'inference': predictions, 'accuracy': acc, 'regularizer' : reg, 'ws': ws, 'ind_inference' : p_ind}
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    adv_model, epsilon = make_adversarial_model(model, fgsm, x, y)
    ph_dict = {'x': x, 'y': y, 'epsilon': epsilon}
    addons = [GlobalStep(),
                TrackAverages(), #do this before train (why?)
                Train(lambda gs: tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate, #0.1
                                                            rho=0.95,
                                                            epsilon=1e-08), FLAGS.batch_size, train_feed={'epsilon' : FLAGS.epsilon}, loss = 'combined_loss', print_steps=100),
                Histograms(), #includes gradients, so has to be done after train
                Saver(save_steps = 1000, checkpoint_path = 'model.ckpt'),
                SummaryWriter(summary_steps = 100, feed_dict = {'keep_prob': 1.0}),
                Logger(),
                Eval(test_data, FLAGS.batch_size, ['accuracy'], eval_feed={}, eval_steps = 1000, name="test (real)"),
                Eval(test_data, FLAGS.batch_size, ['adv_accuracy'], eval_feed={'epsilon': FLAGS.epsilon}, eval_steps = 1000, name="test (adversarial)")]
                # + evals
    #pl_dict, model = adv_mnist_fs()
    trainer = Trainer(adv_model, FLAGS.max_steps, train_data, addons, ph_dict, train_dir = FLAGS.train_dir, verbosity=1, sess=sess)
    trainer.init_and_train()

if __name__=='__main__':
    app.run()

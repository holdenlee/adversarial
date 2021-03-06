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

def fgsm_clip(x, predictions, eps):
    return fgsm(x,predictions,eps,0,1)

def mix_model(t=100, many_files=True, load_from=None, adv = fgsm, reg_weight=1.0, batch_size=100, verbosity=1):
    model = cnn_model
    models = []
    for i in range(t):
        models.append(model())
    if many_files:
        for i in range(t):
            load_model_t(load_from, models[i], i)
            printv('Loaded model %d' % i, verbosity, 1)
    def model(x, y):
        predictions, ws, p_ind = mix(x, models, batch_size)
        loss = mix_loss(y, predictions)
        loss = tf.identity(loss, name="loss")
        reg = reg_weight * entropy_reg(ws, 0.00001)
        reg = tf.identity(reg, name="regularizer")
        acc = accuracy2(y, predictions)
        tf.add_to_collection('losses', loss)
        tf.add_to_collection('losses', reg)
        #added for comparing accuracies. This is batch_size * t
        #https://www.tensorflow.org/performance/xla/broadcasting
        #batch_size, t* batch_size. Must be this order to broadcast correctly
        ind_correct = tf.cast(tf.transpose(tf.equal(tf.argmax(y,1), tf.transpose(tf.argmax(p_ind,1)))), tf.float32)
        #tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return {'loss': loss, 'inference': predictions, 'accuracy': acc, 'regularizer' : reg, 'ws': ws, 'ind_inference' : p_ind, 'ind_correct' : ind_correct}
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    adv_model, epsilon = make_adversarial_model(model, adv, x, y)
    ph_dict = {'x': x, 'y': y, 'epsilon': epsilon}
    return adv_model, ph_dict, epsilon

def make_evals(test_data, batch_size=100, eval_steps=600, ep_increment = 0.1, ep_range = 5):
    return [Eval(test_data, batch_size, ['adv_accuracy'], eval_feed={'epsilon': i*ep_increment}, eval_steps = eval_steps, name="test (adversarial %f)" % (i*ep_increment)) for i in range(1,ep_range+1)]

#def name_train_dir():
#    return "mix%d_pretrain1_epochs%d_ep%f_reg%f" % (FLAGS.t, FLAGS.max_steps//600, FLAGS.reg_weight)

def make_trainer(X_train, Y_train, X_test, Y_test, adv='fgsm', t=100, load='T', load_from='pretrain/nets', batch_size=100, learning_rate=0.1, save_steps = 1200, summary_steps=100, train_dir='train/', eval_steps=600, print_steps = 100, epsilon=0.3, max_steps = 60000, reg_weight=1, verbosity=1):
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    #train_data = make_batch_feeder({'x': X_train, 'y':Y_train}, lambda: 0.5 * random())
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test}) #don't need to shuffle here
    # https://github.com/fchollet/keras/issues/2310
    #K._LEARNING_PHASE = tf.constant(0)
    K.set_learning_phase(1)
    sess = tf.Session()
    keras.backend.set_session(sess)
    if adv == 'fgsm':
        adv = fgsm
    elif adv == 'fgm':
        adv = fgm
    elif adv == 'fgsm_clip':
        adv = fgsm_clip
    elif adv == 'fgm_clip':
        adv = fgm_clip
    else:
        pass
        #print("Invalid adv. Defaulting to fgsm")
        #adv = fgsm
    adv_model, ph_dict, epsilon1 = mix_model(t=t, many_files=(load=='T'), load_from=load_from, reg_weight=reg_weight, batch_size = batch_size)
    evals = make_evals(test_data, batch_size=batch_size, eval_steps=600, ep_increment = 0.1, ep_range = 5)
    #evals = [Eval(test_data, FLAGS.batch_size, ['adv_accuracy'], eval_feed={'epsilon': i*0.1}, eval_steps = 1000, name="test (adversarial @ %f)" % (i*0.1)) for i in range(1,6)]
    addons = [GlobalStep(),
                TrackAverages(), #do this before train (why?)
                Train(lambda gs: tf.train.AdadeltaOptimizer(learning_rate=learning_rate, #0.1
                                                            rho=0.95,
                                                            epsilon=1e-08), batch_size, train_feed={'epsilon' : epsilon}, loss = 'combined_loss', print_steps=print_steps),
                Histograms(), #includes gradients, so has to be done after train
                Saver(save_steps = save_steps, checkpoint_path = os.path.join(train_dir, 'model.ckpt')),
                SummaryWriter(summary_steps = summary_steps, feed_dict = {}),
                Logger(),
                Eval(test_data, batch_size, ['accuracy'], eval_feed={}, eval_steps = eval_steps, name="test (real)")] + evals
                #Eval(test_data, FLAGS.batch_size, ['adv_accuracy'], eval_feed={'epsilon': FLAGS.epsilon}, eval_steps = FLAGS.eval_steps, name="test (adversarial)")] 
                # + evals
    #pl_dict, model = adv_mnist_fs()
    trainer = Trainer(adv_model, max_steps, train_data, addons, ph_dict, train_dir = train_dir, verbosity=verbosity, sess=sess)
    return trainer


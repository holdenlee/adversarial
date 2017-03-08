import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

from mix import *
#from mnist_model import *
from train_many import *
from train_adv import *

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/train_mix', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mix.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 100, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_string('load', 'T', 'Whether to load nets')
flags.DEFINE_string('load_from', 'pretrain/nets', 'Whether to run control experiment')
flags.DEFINE_integer('t', 100, 'Number of nets')
flags.DEFINE_integer('label_smooth', 0.1, 'How much to clip y values (0 for no clipping)')
flags.DEFINE_float('reg_weight', 1, 'Weight on entropy regularizer')
flags.DEFINE_float('epsilon', 0.3, 'Strength of attack')
#flags.DEFINE_string('control', 'T', 'Whether to run control experiment')

# NOTE: because batch must be specified explicitly for the matrix multiplication (terrible!) the batch_size must divide train/test size (60000, 10000). 
def main(argv=None):
    #tf.reset_default_graph()
    sess = tf.Session()
    keras.backend.set_session(sess)
    X_train, Y_train, X_test, Y_test = data_mnist()

    assert Y_train.shape[1] == 10.
    Y_train = Y_train.clip(FLAGS.label_smooth / 9., 1. - FLAGS.label_smooth)

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    model = cnn_model
    models = 100*[model()]
    if FLAGS.load == 'T':
        for i in range(FLAGS.t):
            load_model_t(FLAGS.load_from, models[i], i)
    predictions, ws = mix(x, models, FLAGS.batch_size)
    #loss = lambda pred: mix_loss(y, pred)
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    model_train_adv(lambda x: mix(x, models, FLAGS.batch_size), sess, x, y, X_train, Y_train, X_test, Y_test, evaluate_adv, train_params, FLAGS.batch_size, fgsm, FLAGS.epsilon, 
        model_loss_fn = lambda pred: mix_loss(y, pred), 
                    reg_fn = lambda x, y: FLAGS.reg_weight*mix_reg(x,y))


if __name__ == '__main__':
    app.run()

    
"""
nohup python train_mix.py --t=100 > train_mix/log_2017-3-7.txt 2>&1 &
#train_mix100_pretrain1_epochs100_ep0.3_reg1
nohup python train_mix.py --t=100 --nb_epochs=100 > train_mix100_pretrain1_epochs100_ep0.3_reg1/log.txt 2>&1 &
"""

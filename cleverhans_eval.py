import tensorflow as tf
from tensorflow.python.platform import flags
from functools import *

import os.path
import numpy as np

from utils import *
from tf_vars import *
from tf_utils import *
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from adv_model import *
from mwu import *
from mwu_adv import *

from mnist_utils import *

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, 'Size of training batches')
flags.DEFINE_float('epsilon', 0.3, 'Strength of attack')
flags.DEFINE_string('adv_f', 'fgsm', 'Type of attack')

def evaluate(sess, x, y, predictions, predictions_adv, X_test, Y_test, batch_size=128):
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print('Test accuracy on legitimate test examples: ' + str(accuracy))
    # Evaluate the accuracy of the adversarially trained MNIST model on
    # adversarial examples
    accuracy_adv = model_eval(sess, x, y, predictions_adv, X_test,
                              Y_test, args=eval_params)
    print('Test accuracy on adversarial examples: ' + str(accuracy_adv))
    return accuracy, accuracy_adv

#example: model=model1_logits()
def cleverhans_evaluate(model_f, filepath, adv_f, epsilon, batch_size=128):
    tf.reset_default_graph()
    # Create TF session and set as Keras backend session
    sess = start_keras_session(1)
    
    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = get_data_mnist()

    # Define input TF placeholder
    x, y = make_phs()
    
    model = model_f()
    predictions = model(x)
    adv_x = adv_f(x, predictions, eps=epsilon)
    predictions_adv = model(adv_x)
    saver = tf.train.Saver()
    #with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, filepath)
    print("Model restored.")
    return evaluate(sess, x, y, predictions, predictions_adv, X_test, Y_test, batch_size=batch_size)
                    
if __name__ == '__main__':
    adv_f = str_to_attack(FLAGS.adv_f)
    cleverhans_evaluate(model1_logits, 
                        tf.train.latest_checkpoint('./single_epochs6_ep0.3/'), 
                        adv_f,
                        FLAGS.epsilon,
                        FLAGS.batch_size)
    cleverhans_evaluate(lambda: mnist_mwu_model(u=20,u2=20), 
                        tf.train.latest_checkpoint('./train_adv_mwu/'), 
                        adv_f,
                        FLAGS.epsilon,
                        FLAGS.batch_size)


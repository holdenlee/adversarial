import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

from mnist_model import *

"""
FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_string('control', 'T', 'Whether to run control experiment')

default_train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
"""
def evaluate(sess, x, y, predictions, X_test, Y_test, batch_size = 128):
    def f():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))
        return accuracy
    return f

def evaluate_adv(sess, x, y, predictions, predictions_adv, X_test, Y_test, batch_size = 128):
    # Evaluate the accuracy of the adversarialy trained MNIST model on
    # legitimate test examples
    def f():
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))
        # Evaluate the accuracy of the adversarially trained MNIST model on
        # adversarial examples
        accuracy_adv = model_eval(sess, x, y, predictions_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: ' + str(accuracy_adv))
    return f

#model can return multiple things. The first thing it returns is the predictions.
def model_train_adv(model, sess, x, y, X_train, Y_train, X_test, Y_test, evaluate_adv1, train_params, batch_size = 128, adv = fgsm, eps=0.3, model_loss_fn = None, reg_fn = None):
    li = model(x)
    predictions = li[0] if isinstance(li, (list, tuple)) else li
    if reg_fn == None:
        r = 0
    else:
        r = reg_fn(*li) if isinstance(li, (list, tuple)) else reg_fn(li)
    adv_x = adv(x, predictions, eps)
    li_adv = model(adv_x)
    predictions_adv = li_adv[0] if isinstance(li_adv, (list, tuple)) else li_adv
    model_train(sess, x, y, predictions, X_train, Y_train,
                predictions_adv=predictions_adv, evaluate=evaluate_adv1(sess, x, y, predictions, predictions_adv, X_test, Y_test, batch_size = batch_size),
                args=train_params, model_loss_fn = model_loss_fn, reg = r)
    #SANITY CHECK
        #model, sess, x, y, X_test, Y_test, evaluate_adv1, train_params, batch_size = FLAGS.batch_size, adv = fgsm, eps=0.3
    #sanity check
    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    eval_params = {'batch_size': batch_size}
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)
    assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test,
                          args=eval_params)
    print('Test accuracy on adversarial examples: ' + str(accuracy))

"""
if __name__ == '__main__':
    sess = tf.Session()
    #keras.backend.set_session(sess)

    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
\    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    predictions = inference(x, 1) # don't do dropout

    # Train an MNIST model
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    adv_x = fgsm(x, predictions, eps=0.3)
    predictions_adv = inference(adv_x, 1)
    if FLAGS.control == 'T':
        model_train(sess, x, y, predictions, X_train, Y_train,
                    evaluate=evaluate_adv(sess, x, y, predictions, predictions_adv, X_test, Y_test, batch_size = FLAGS.batch_size), args=train_params)
        # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
        #adv_x = fgsm(x, predictions, eps=0.3)
        eval_params = {'batch_size': FLAGS.batch_size}
        X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)
        assert X_test_adv.shape[0] == 10000, X_test_adv.shape
        accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test,
                              args=eval_params)
        print('Test accuracy on adversarial examples: ' + str(accuracy))

    #(model, sess, x, y, X_train, Y_train, X_test, Y_test, evaluate_adv1, train_params, batch_size = FLAGS.batch_size, adv = fgsm, eps=0.5)
    #RESET
    tf.reset_default_graph()
    sess = tf.Session()
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    #epochs passed as flag?
    model_train_adv(lambda x1: inference(x1,1), sess, x, y, X_train, Y_train, X_test, Y_test, evaluate_adv, train_params, FLAGS.batch_size, fgsm, 0.5) 


def fg(y, x):
    # http://stackoverflow.com/questions/35226428/how-do-i-get-the-gradient-of-the-loss-at-a-tensorflow-variable
    grad = tf.gradients(y, x)
    #https://www.tensorflow.org/api_docs/python/tf/norm
    ngrad = grad/tf.norm(grad)
    return x + ep * grad
"""
#def fgs(y, x):

#do a stop-gradient?

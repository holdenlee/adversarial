from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')

sess = tf.Session()


# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))

model_mnist = cnn_model

# Define TF model graph
model = model_mnist()
predictions = model(x)
print("Defined TensorFlow model graph.")

# Get MNIST test data
X_train, Y_train, X_test, Y_test = data_mnist()

# Train an MNIST model
model_train(sess, x, y, predictions, X_train, Y_train)

# Evaluate the accuracy of the MNIST model on legitimate test examples
accuracy = model_eval(sess, x, y, predictions, X_test, Y_test)
assert X_test.shape[0] == 10000, X_test.shape
print('Test accuracy on legitimate test examples: ' + str(accuracy))

# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
adv_x = fgsm(x, predictions, eps=0.3)
X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])
assert X_test_adv.shape[0] == 10000, X_test_adv.shape

# Evaluate the accuracy of the MNIST model on adversarial examples
accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test)
print('Test accuracy on adversarial examples: ' + str(accuracy))

# Redefine TF model graph
model_2 = model_mnist()
predictions_2 = model_2(x)
adv_x_2 = fgsm(x, predictions_2, eps=0.3)
predictions_2_adv = model_2(adv_x_2)

# Perform adversarial training
model_train(sess, x, y, predictions_2, X_train, Y_train, predictions_adv=predictions_2_adv)

# Evaluate the accuracy of the adversarialy trained MNIST model on
# legitimate test examples
accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test)
print('Test accuracy on legitimate test examples: ' + str(accuracy))

# Craft adversarial examples using Fast Gradient Sign Method (FGSM) on
# the new model, which was trained using adversarial training
X_test_adv_2, = batch_eval(sess, [x], [adv_x_2], [X_test])
assert X_test_adv_2.shape[0] == 10000, X_test_adv_2.shape

# Evaluate the accuracy of the adversarially trained MNIST model on
# adversarial examples
accuracy_adv = model_eval(sess, x, y, predictions_2, X_test_adv_2, Y_test)
print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

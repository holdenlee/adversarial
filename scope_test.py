import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

from mnist_model import *

FLAGS = flags.FLAGS

def test_tf_slim(x):
    slim.bias_add(initializer=init_ops.ones_initializer())

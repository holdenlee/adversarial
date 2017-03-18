import keras as K

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils import cnn_model
from cleverhans.utils_mnist import data_mnist

from tf_utils import *
from train_many import *

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    #K.set_learning_phase(1)
    #sess = tf.Session()
    #keras.backend.set_session(sess)
    model = cnn_model()
    #load_model_t('pretrain/nets', model, 0)
    load_model_t('tmp/nets', model, 0)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=128)
    #print(model.get_weights())
    print(loss_and_metrics)

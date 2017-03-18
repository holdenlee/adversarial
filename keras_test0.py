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
    model = cnn_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=1, batch_size=32)
    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=128)
    #print(model.get_weights())
    print(loss_and_metrics)
    #model.save_weights('tmp.hdf5')
    save_model_t('tmp', model, 0)

    #K.set_learning_phase(1)
    #sess = tf.Session()
    #keras.backend.set_session(sess)
    
    model2 = cnn_model()
    load_model_t('tmp', model2, 0)
    #model2.load_weights('tmp.hdf5')
    model2.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    loss_and_metrics = model2.evaluate(X_test, Y_test, batch_size=128)
    #print(model.get_weights())
    print(loss_and_metrics)

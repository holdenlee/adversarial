import keras as K

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils import cnn_model
from cleverhans.utils_mnist import data_mnist

from tf_utils import *
from train_many import *
from mix import *

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = data_mnist()
    #assert Y_train.shape[1] == 10.
    #Y_train = Y_train.clip(FLAGS.label_smooth / 9., 1. - FLAGS.label_smooth)
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    #train_data = make_batch_feeder({'x': X_train, 'y':Y_train}, lambda: 0.5 * random())
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test}) #don't need to shuffle here
    K.backend.set_learning_phase(1)
    sess = tf.Session()
    keras.backend.set_session(sess)
    model = cnn_model()
    print("AAA")
    print(model.get_weights())
    load_model_t('pretrain/nets', model, 0)
    print("BBB")
    print(model.get_weights())
    def model2(x, y):
        predictions = model(x)
        #predictions, ws, p_ind = mix(x, models, FLAGS.batch_size)
        loss = mix_loss(y, predictions)
        loss = tf.identity(loss, name="loss")
        #reg = FLAGS.reg_weight * entropy_reg(ws, 0.00001)
        #reg = tf.identity(reg, name="regularizer")
        acc = accuracy2(y, predictions)
        tf.add_to_collection('losses', loss)
        #tf.add_to_collection('losses', reg)
        return {'loss': loss, 'inference': predictions, 'accuracy': acc}
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    #adv_model, epsilon = make_adversarial_model(model, fgsm, x, y)
    ph_dict = {'x': x, 'y': y} #, 'epsilon': epsilon}
    m = model2(x,y)
    """
    addons = [GlobalStep(),
                TrackAverages(), #do this before train (why?)
                Train(lambda gs: tf.train.AdadeltaOptimizer(learning_rate=0.1, #0.1
                                                            rho=0.95,
                                                            epsilon=1e-08), 100, loss = 'loss', print_steps=100),
                Histograms(), #includes gradients, so has to be done after train
                Saver(save_steps = 1000, checkpoint_path = 'model.ckpt'),
                SummaryWriter(summary_steps = 100),
                #Logger(),
                Eval(test_data, 100, ['accuracy'], eval_feed={}, eval_steps = 1, name="test (real)")]
                # + evals
    #pl_dict, model = adv_mnist_fs()
    """
    addons = [Eval(test_data, 100, ['accuracy'], eval_feed={}, eval_steps = 1, name="test (real)")]
    trainer = Trainer(m, 2, train_data, addons, ph_dict, train_dir = 'tmp', verbosity=1, sess=sess)
    trainer.init_and_train()
    print("CCC")
    print(model.get_weights())
    print("DDD")
    print(trainer.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))



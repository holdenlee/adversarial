import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

#from mnist_model import *
from mnist_models import *

import os.path

import keras
from keras import backend

from train_many import *
from mix import *
from train_mix import *
from utils import *
from PIL import Image

def cnn_model_summary():
    model = cnn_model()
    print(model.summary())

def examine_filters(m1, layer=0):
    #this is 8x8x1x64
    filters = m1.layers[layer].get_weights()[0]
    a = np.reshape(filters, [64,64]) #the columns are the weights for a single filter
    dots = np.dot(np.transpose(a), a)
    print(dots)
    d_inv_sqrt = np.diag(1/np.sqrt(np.diagonal(dots)))
    normed_dots = np.dot(np.dot(d_inv_sqrt, dots), d_inv_sqrt)
    print("normed:")
    print(normed_dots)
    return dots, normed_dots

def save_filters(m1, filename, layer=0):
    #this is 8x8x1x64
    filters = m1.layers[layer].get_weights()[0]
    filters = np.transpose(filters, (3, 0, 1, 2))
    filters = filters.reshape((64,8,8))
    for i in range(64):
        save_image(filters[i], filename+str(i), shape=(8,8))

def save_image(array, name, shape = (28,28)):
    img = Image.fromarray(np.ndarray.astype(255*array.reshape(shape), np.dtype(np.uint8)), mode='L')
    img.save('%s.png' % name)
"""
def save_dict_as_pics(A, x, y, unnorm_f= lambda x:x, filename = 'dict'):
    rows = np.shape(A)[0]
    Ar = np.reshape(A, (rows, x, y))
    for i in range(rows):
        img = Image.fromarray(np.ndarray.astype(unnorm_f(Ar[i]), np.dtype(np.uint8)), 'L')
        img.save('%s_%d.png' % (filename, i))
"""

def save_images(ims, name):
    for (d,array) in enumerate(ims):
        save_image(array, '%s_%d' % (name, d))

def save_adv_images(X, Y, filepath= 'pretrain5_model1_smooth0/nets0.hdf5', save_as = 'images/pretrain5_model1_smooth0_adv', adv = fgsm2_clip, ep=0.3):
    K.set_learning_phase(0)
    sess = tf.Session()
    K.set_session(sess)
    #must do all this before loading model
    model = model1_no_dropout() #BUG
    #model.summary()
    model.load_weights(filepath, by_name=True)
    #print(model.get_weights())
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    #def m(x,y):
    #    loss = mix_loss(y, predictions)
    #    return {'loss': loss, 'inference': predictions, 'accuracy': acc, 'regularizer' : reg, 'ws': ws, 'ind_inference' : p_ind, 'ind_correct' : ind_correct}
    d, epsilon = make_adversarial_model(make_model(model), adv, x, y)
    adv_xs = sess.run((d['adv_x']), feed_dict = {x:X, y:Y, epsilon: ep})
    save_images(adv_xs, save_as)

def get_grad_corr(X, Y, filepath='pretrain5_model1_smooth0/nets', save_as = 'images/pretrain5_model1_smooth0_grad_corr.txt', adv = fgsm2_clip, t=100):
    K.set_learning_phase(0)
    sess = tf.Session()
    K.set_session(sess)
    #must do all this before loading model
    model = model1_no_dropout  
    #cnn_model #BUG with keras
    models = []
    dicts = []
    #eps = []
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    for i in range(t):
        models.append(model())
        load_model_t(filepath, models[i], i)
        print('Loaded model %d' % i)
        d, epsilon = make_adversarial_model(make_model(models[i]), adv, x, y)
        dicts.append(d)
            #eps[i] = epsilon
    #print(model.get_weights())
    dots = []
    avgs = np.zeros((t,t))
    for m in range(len(X)):
        adv_grads=[]
        for i in range(t):
            adv_grad = sess.run((d['adv_grad']), feed_dict = {x:X[i:i+1], y:Y[i:i+1]}).flatten()
            adv_grads.append(adv_grad)
        adv_grads = np.asarray(adv_grads) #rows are gradients
        dots.append(np.dot(adv_grads, np.transpose(adv_grads)))
        avgs = avgs + dots[m]
        print(dots[m])
    avgs = avgs/len(X)
    print("Average:", avgs)
    np.savetxt(save_as, avgs)

if __name__=='__main__':
    np.set_printoptions(threshold=np.nan, precision=5)
    #cnn_model_summary()
    #* Examine filters
    #examine_filters(cnn_model())
    X_train, Y_train, X_test, Y_test = data_mnist()
    sess, models, dicts, eps, x, y = load_many(model1_no_dropout, filepath='pretrain5_model1_smooth0/nets', adv = fgsm2_clip, t=100)
    for (i,m) in enumerate(models):
        print("Model %d" %i)
        examine_filters(m)
    ## is (60000,28,28,1), numbers 0 to 1.
    #save_image(X_train[0], 'test')
    #* Save adversarial images
    """
    save_adv_images(X_test[0:5], Y_test[0:5])
    tf.reset_default_graph()
    save_adv_images(X_test[0:5], Y_test[0:5], filepath='pretrain5_model1_smooth0/nets0.hdf5', save_as = 'images/pretrain5_model1_smooth0_adv0.5', adv = fgsm2_clip, ep=0.5)
    tf.reset_default_graph()
    save_adv_images(X_test[0:5], Y_test[0:5], filepath='pretrain5_model1_smooth0/nets0.hdf5', save_as = 'images/pretrain5_model1_smooth0_adv0.7', adv = fgsm2_clip, ep=0.7)
    """
    #* Get correlations between gradients
    #note: I don't think the training clipped values. does it?
    #get_grad_corr(X_test[0:10], Y_test[0:10], filepath='pretrain5_model1_smooth0/nets', save_as = 'images/pretrain5_model1_smooth0_grad_corr.txt', adv = fgsm2_clip, t=9)

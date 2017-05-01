import tensorflow as tf
from functools import *
import itertools
import inspect
from datetime import datetime
import re
import os
import os.path
import time
import numpy as np
#from six.moves import xrange
from utils import *
from tf_vars import *
from tf_utils import *
from cleverhans.utils_mnist import data_mnist

import operator

def prod(li):
    return reduce(operator.mul, li, 1)

def make_mw_ops(y, ws, eta, grads=None): #include clipping
    # https://www.tensorflow.org/api_docs/python/tf/gradients
    if grads == None:
        grads = tf.gradients(y, ws)
    assign_ops = []
    for (w,g) in zip(ws, grads):
        # https://blog.metaflow.fr/tensorflow-mutating-variables-and-control-flow-2181dd238e62
        w1 = w*(1-eta*g)
        #normalize each column
        w2 = w1 / tf.reduce_sum(w1, range(tf.rank(w1)-1), keep_dims = True)
        w_op = tf.assign(w, w2)
        assign_ops.append(w_op)
    return assign_ops

class MWOptimizer:
    def __init__(self, ws, bs, learning_rate=0.01,smoothing=0):
        self.ws = ws #hand all parameters
        self.bs = bs
        self.grads = None
        self.b_grads = None
        self.eta  = learning_rate
        self.smoothing= smoothing
        #self.global_step= global_step
    def compute_gradients(self, y):
        self.grads = tf.gradients(y,self.ws)
        self.b_grads = tf.gradients(y,self.bs)
        print("computing gradients:"+str((self.grads, self.b_grads)))
        return (self.grads, self.b_grads)
    def apply_gradients(self, gradients, global_step=None):
        #if self.grads == None:
        #    gradients = self.compute_gradients(y)
        return make_mw_ops_with_biases(self.ws, self.bs, self.eta, gradients=gradients, smoothing=self.smoothing, global_step = global_step)

def make_mw_ops_with_biases(ws, bs, eta, y=None, gradients=None, smoothing=0, global_step = None): #include clipping
    # https://www.tensorflow.org/api_docs/python/tf/gradients
    if gradients==None:
        grads = tf.gradients(y, ws)
        b_grads = tf.gradients(y, bs)
    else:
        (grads, b_grads) = gradients
    print('making grads: ', gradients)
    w_assign_ops = []
    b_assign_ops = []
    if global_step !=None:
        g_assign = [tf.assign(global_step, global_step+1)]
    else:
        g_assign = []
    for (w,g,b,gb) in zip(ws, grads, bs, b_grads):
        # https://blog.metaflow.fr/tensorflow-mutating-variables-and-control-flow-2181dd238e62
        #g = tf.Print(g,[g],'grad W:',summarize=128)
        w1 = w*tf.exp(-eta*g)
        #print('making grads: ', b, eta, gb)
        b1 = b*tf.exp(-eta*gb)
        ld = int(w1.get_shape()[-1])
        d = prod([int(i) for i in w1.get_shape()[:-1]])
        #normalize each column
        z = tf.reduce_sum(w1, axis = list(range(len(w1.get_shape())-1))) + b1[0:ld] + b[ld:2*ld]
                          #range(tf.rank(w1)-1)) + b1
        w2 = (w1 / z) * (1-smoothing) + (1/(d+2*ld))*smoothing 
        b2 = (b1 / tf.tile(z,[2])) * (1-smoothing) + (1/(d+2*ld))*smoothing
        with tf.control_dependencies(g_assign):
            w_op = tf.assign(w, w2)
            b_op = tf.assign(b,b2)
        w_assign_ops.append(w_op)
        b_assign_ops.append(b_op)
    with tf.control_dependencies(w_assign_ops+b_assign_ops):
        op = tf.no_op()
    return op

def pm(x):
    return tf.concat([x, -x], axis=-1)

#http://stackoverflow.com/questions/29831489/numpy-1-hot-array

def np_onehot(i, n):
    b = np.zeros([n])
    b[i] = 1
    return b
    #l = np.size(a)
    #b = np.zeros([l,n])
    #b[np.arange(l), a] = 1
    #return b


def synthesize_data(bf, batch_size):
    [dim, i, sgn] = bf.args
    xs = 2*np.random.randint(0,2, size = (batch_size, dim)) - 1
    ys = np.asarray([np_onehot(int((sgn * x[i] + 1)/2),2) for x in xs])
    #print("ys shape", ys.shape)
    return {'x': xs, 'y': ys}

def mnist_linear(x,y,dim=784, n=10, labels=False,u=1):
    x1 = tf.reshape(x, (-1, dim))
    pm_x = pm(x1)
    W = get_scope_variable('W', shape=[2*dim, n], initializer = tf.constant_initializer(1/(2*dim+1)))
    b = get_scope_variable('b', shape=[2*n], initializer = tf.constant_initializer(1/(2*dim+1)))
    #tf.ones([2*dim,10])/(2*dim))
    yhat = u*(tf.matmul(pm_x, W) + b[0:n] - b[n:2*n])
    #yhat = tf.nn.softmax(u*(tf.matmul(pm_x, W) + b))
    #print(y.get_shape())
    #print(yhat.get_shape())
    loss = tf.losses.softmax_cross_entropy(y, yhat)
    acc = accuracy(y,yhat) if labels else accuracy2(y, yhat)
    return {'loss': loss, 'inference' : yhat, 'accuracy': acc}, W, b

def dirichlet(dims,n):
    s = prod(dims)
    return np.reshape(np.random.dirichlet(np.ones([s]), n), [n]+dims)

def dirichlet_initializer(dims,n):
    s = prod(dims)
    r = np.random.dirichlet(np.ones([s+2]), n).transpose().astype(np.float32)
    print(r[0,0:10])
    return np.reshape(r[0:s,:], dims+[n]), np.ndarray.flatten(r[s:s+2,:])

def mwu_linear_layer(x, n, u=50, name='fc'):
    pm_x = pm(x)
    dim = int(pm_x.get_shape()[-1])
    #print('dim', dim)
    W0,b0 = dirichlet_initializer([dim],n)
    W = get_scope_variable('W'+name, initializer = W0, dtype = 'float32')
                           #shape=[dim, n], tf.constant_initializer(1/(dim+2)))
    b = get_scope_variable('b'+name, initializer = b0, dtype = 'float32')
    #shape=[2*n], 
    #print(b[0:n].get_shape())
    #print(tf.matmul(pm_x, W).get_shape())
    y = (u*(tf.matmul(pm_x, W) + b[0:n] - b[n:2*n]))
    #tf.nn.softmax
    return y, W, b

def make_softmax_model(y,yhat, labels=False):
    loss = tf.losses.softmax_cross_entropy(y, yhat)
    acc = accuracy(y,yhat) if labels else accuracy2(y, yhat)
    return {'loss': loss, 'inference' : yhat, 'accuracy': acc}

def mwu_conv_layer(x, dims, u=50, padding = 'valid', strides = (1,1), name='', f = tf.nn.relu):
                   #tf.nn.relu):
    x1 = pm(x)
    [a,b,c,d] = dims
    dims2 = [a, b, 2*c, d]
    W0,b0 = dirichlet_initializer([a,b,2*c],d)
    W = get_scope_variable('W'+name, initializer = W0)
                           #shape = dims2, initializer = tf.constant_initializer(1/(2*a*b*c+2)))
    bias = get_scope_variable('b'+name, initializer = b0)
                              #shape = [2*d], initializer = tf.constant_initializer(1/(2*a*b*c+2)))
    #print(bias[0:d].get_shape())
    #print(tf.nn.convolution(x1, W, padding, strides=strides).get_shape())
    y = f(u*(tf.nn.convolution(x1, W, padding, strides=strides)  + bias[0:d] - bias[d:2*d])) #, dilation_rate=None, name=None, data_format=None)
    tf.add_to_collection('Ws', W)
    tf.add_to_collection('bs', bias)
    return y, W, bias

def mnist_conv(x,y, u=50,u2=50):
    return mnist_conv_u(x,y,u,u,u,u2)

def mnist_conv_u(x,y, u1=50,u2=50, u3=50,u4=50):
    y1, W1, b1 = mwu_conv_layer(x,[8, 8, 1, 64], u=u1, name='1', strides=(2,2), padding = 'SAME')
    y2, W2, b2 = mwu_conv_layer(y1,[6, 6, 64, 128], u=u2, name='2', strides=(2,2), padding = 'VALID')
    y3, W3, b3 = mwu_conv_layer(y2,[5, 5, 128, 128], u=u3, name='3', padding = 'VALID')
    yf = tf.contrib.layers.flatten(y3)
    y4, W4, b4 = mwu_linear_layer(yf, 10, name='4', u=u4)
    #y4 = tf.Print(y4, [tf.nn.softmax(y4)], "output", summarize=10)
    model = make_softmax_model(y,y4)
    return model, [W1, W2, W4], [b1, b2, b4]

def mnist_conv1(x,y, u=50,u2=50):
    y1, W1, b1 = mwu_conv_layer(x,[8, 8, 1, 64], u=u, name='1', strides=(2,2), padding = 'SAME')
    yf = tf.contrib.layers.flatten(y1)
    y4, W4, b4 = mwu_linear_layer(yf, 10, name='4',u=u2)
    #y4 = tf.Print(y4, [tf.nn.softmax(y4)], "output", summarize=10)
    model = make_softmax_model(y,y4)
    return model, [W1, W4], [b1, b4]

"""
def mnist_linear0(x,y):
    x1 = tf.reshape(x, (-1, 784))
    pm_x = pm(x1)
    dim = 784
    W = get_scope_variable('W', shape=[2*dim, 10], initializer = tf.constant_initializer(1/(2*dim+1)))
    b = get_scope_variable('b', shape=[10], initializer = tf.constant_initializer(1/(2*dim+1)))
    #tf.ones([2*dim,10])/(2*dim))
    yhat = tf.nn.softmax(tf.matmul(pm_x, W) + b)
    loss = tf.losses.softmax_cross_entropy(y, yhat)
    acc = accuracy2(y, yhat)
    return {'loss': loss, 'inference' : yhat, 'accuracy': acc}, W, b
"""

def basic_test(dim=100, print_steps=1, max_steps=10, batch_size=128):
    i = np.random.randint(0, dim)
    sgn = 2*np.random.randint(0,2)-1
    bf = BatchFeeder([dim, i, sgn], 100, synthesize_data)
    x = tf.placeholder(tf.float32, shape=(None, dim))
    y = tf.placeholder(tf.float32, shape=(None, 2))
    model, W, b = mnist_linear(x,y, dim,n=2, labels=False)
    ph_dict = {'x': x, 'y': y}
    evaler = Eval(bf, 100, ['accuracy'], eval_steps = print_steps, name='test')
    opter= lambda: MWOptimizer([W], [b], learning_rate=0.1)
    addons = [Train(opter, 
                    batch_size, 
                    train_feed={}, 
                    loss = 'loss', 
                    print_steps=print_steps),
              evaler]
    trainer = Trainer(model, max_steps, bf, addons, ph_dict, train_dir = 'pm1/', verbosity=1)
    trainer.init()
    print("Weights:",trainer.sess.run([W,b]))
    trainer.train()
    [weights, biases] = trainer.sess.run([W,b])
    print("Weights:",weights, biases)
    print(np.sum(weights, axis=0))
    print(np.max(weights, axis=0))
    trainer.finish()
    return evaler.record[-1][0]

#def pm_ones(x):

def mnist_linear_test(u=50, lr=0.1,max_steps=1200, smoothing=0.01):
    X_train, Y_train, X_test, Y_test = data_mnist()
    X_train =2*X_train-1
    X_test =2*X_test-1
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test})
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    model, W, b = mnist_linear(x,y,u=50)
    #model, Ws, bs = mnist_conv(x,y,u=50)
    Ws=[W]
    bs = [b]
    ph_dict = {'x': x, 'y': y}
    return make_mwu_trainer(model, ph_dict, Ws, bs, train_data, test_data, lr=lr,max_steps=max_steps, smoothing=smoothing)

def mnist_mwu_model(u=50,u2=50):
    def f(x):
        #print('0:'+str(tf.shape(x)))
        #print('0:'+str(x.get_shape()))
        y1, W1, b1 = mwu_conv_layer(x,[8, 8, 1, 64], u=u, name='1', strides=(2,2), padding = 'SAME')
        #print('1:'+str(tf.shape(y1)))
        #print('1:'+str(y1.get_shape()))
        y2, W2, b2 = mwu_conv_layer(y1,[6, 6, 64, 128], u=u, name='2', strides=(2,2), padding = 'VALID')
        #print('2:'+str(tf.shape(y2)))
        #print('2:'+str(y2.get_shape()))
        y3, W3, b3 = mwu_conv_layer(y2,[5, 5, 128, 128], u=u, name='3', padding = 'VALID')
        #print('3:'+str(tf.shape(y3)))
        #print('3:'+str(y3.get_shape()))
        #dim = tf.reduce_prod(tf.shape(y3)[1:])
        yf = tf.reshape(y3, [-1, 128])
        #http://stackoverflow.com/questions/36668542/flatten-batch-in-tensorflow
        #yf = tf.contrib.layers.flatten(y3)
        y4, W4, b4 = mwu_linear_layer(yf, 10, name='4', u=u2)
        y = tf.nn.softmax(y4)
        return y
    return f


def make_mwu_trainer(model, ph_dict, Ws, bs, train_data, test_data, lr=0.1,max_steps=3000, smoothing=0.01,batch_size = 100, print_steps = 100, eval_steps = 600, verbosity=1, train_dir = "mwu/"):
    #opter = lambda: tf.train.GradientDescentOptimizer(learning_rate=0.1)
    opter= lambda: MWOptimizer(Ws, bs, learning_rate=lr, smoothing=smoothing)
    evaler = Eval(test_data, batch_size, ['accuracy'], eval_feed={}, eval_steps = eval_steps, name="test")
    addons = [Train(opter, 
                    batch_size, 
                    train_feed={}, 
                    loss = 'loss', 
                    print_steps=print_steps),
              evaler]
    trainer = Trainer(model, max_steps, train_data, addons, ph_dict, train_dir = train_dir, verbosity=verbosity)
    trainer.init()
    #print("Weights:",trainer.sess.run([W,b]))
    trainer.train()
    #[weights, biases] = trainer.sess.run([W,b])
    #print("Weights:",weights, biases)
    #print(np.sum(weights, axis=0))
    #print(np.max(weights, axis=0))
    trainer.finish()
    try:
        ans = evaler.record[-1][0]
    except IndexError:
        ans = 0
    return ans

def make_mwu_trainer_gs(model, ph_dict, Ws, bs, train_data, test_data, lr=0.1,max_steps=3000, smoothing=0.01,batch_size = 100, print_steps = 100, eval_steps = 600, verbosity=1, train_dir = "mwu/"):
    #opter = lambda: tf.train.GradientDescentOptimizer(learning_rate=0.1)
    opter= lambda gs: MWOptimizer(Ws, bs, learning_rate=lr/tf.ceil(tf.cast(gs+1, tf.float32)/500), smoothing=smoothing)
    # /tf.ceil(tf.cast(gs+1, tf.float32)/1000)
    evaler = Eval(test_data, batch_size, ['accuracy'], eval_feed={}, eval_steps = eval_steps, name="test")
    addons = [GlobalStep(),
              Train(opter, 
                    batch_size, 
                    train_feed={}, 
                    loss = 'loss', 
                    print_steps=print_steps),
              evaler]
    trainer = Trainer(model, max_steps, train_data, addons, ph_dict, train_dir = train_dir, verbosity=verbosity)
    trainer.init()
    #print("Weights:",trainer.sess.run([W,b]))
    trainer.train()
    #[weights, biases] = trainer.sess.run([W,b])
    #print("Weights:",weights, biases)
    #print(np.sum(weights, axis=0))
    #print(np.max(weights, axis=0))
    trainer.finish()
    try:
        ans = evaler.record[-1][0]
    except IndexError:
        ans = 0
    return ans

def mnist_conv_test(u=50, u2=50, lr=0.1,max_steps=3000, smoothing=0.01):
    print(u,u2,lr)
    X_train, Y_train, X_test, Y_test = data_mnist()
    X_train =2*X_train-1
    X_test =2*X_test-1
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test})
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    #model, W, b = mnist_linear(x,y,u=u)
    model, Ws, bs = mnist_conv(x,y,u=u,u2=u2)
    ph_dict = {'x': x, 'y': y}
    return make_mwu_trainer(model, ph_dict, Ws, bs, train_data, test_data, lr=lr,max_steps=max_steps, smoothing=smoothing)

def mnist_conv_test_gs(u=50, u2=50, lr=0.1,max_steps=3000, smoothing=0.01):
    print(u,u2,lr)
    X_train, Y_train, X_test, Y_test = data_mnist()
    X_train =2*X_train-1
    X_test =2*X_test-1
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test})
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    #model, W, b = mnist_linear(x,y,u=u)
    model, Ws, bs = mnist_conv(x,y,u=u,u2=u2)
    ph_dict = {'x': x, 'y': y}
    return make_mwu_trainer_gs(model, ph_dict, Ws, bs, train_data, test_data, lr=lr,max_steps=max_steps, smoothing=smoothing)

def mnist_conv_test2(u=50, u2=50, lr=0.1,max_steps=3000, smoothing=0.01):
    print(u,u2,lr)
    X_train, Y_train, X_test, Y_test = data_mnist()
    X_train =2*X_train-1
    X_test =2*X_test-1
    train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
    test_data = make_batch_feeder({'x': X_test, 'y':Y_test})
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    u1 = tf.placeholder(tf.float32, shape=1)
    u2 = tf.placeholder(tf.float32, shape=1)
    u3 = tf.placeholder(tf.float32, shape=1)
    u4 = tf.placeholder(tf.float32, shape=1)
    #model, W, b = mnist_linear(x,y,u=u)
    model, Ws, bs = mnist_conv(x,y,u=u,u2=u2)
    ph_dict = {'x': x, 'y': y}
    make_mwu_trainer(model, ph_dict, Ws, bs, train_data, test_data, lr=lr,max_steps=max_steps, smoothing=smoothing)

if __name__=='__main__':
    np.set_printoptions(threshold=np.inf, precision=3)
    #basic_test()
    #ans = fors_zip([[20,50,80], [0.1, 0.01]], 
    #               lambda li: mnist_linear_test(li[0], li[1], smoothing=0))
    #ans = fors_zip([[20],[20],[0.3,0.1,0.03,0.01]],
    #    [[10, 20, 50], [10, 20, 50], [0.3, 0.1, 0.03, 0.01]],
        #[[1, 3, 10, 20 ,50],[1, 3, 10, 20 ,50],[0.1]],
        #[[1, 3, 10, 20 ,50],[1, 3, 10, 20 ,50],[0.1]], 
    #               lambda li: mnist_conv_test(li[0], li[1], li[2], 3000, smoothing=0))
    #ans = fors_zip([[20],[20],[0.1],[0,0.001,0.01,0.1]],
    #               lambda li: mnist_conv_test(li[0], li[1], li[2], 3000, smoothing=li[3]))
    ans = mnist_conv_test(20,20,0.3, 10000,smoothing=0.00)
    print(ans)
    #np.savetxt(os.path.join('mwu', 'conv_test.txt'), ans)

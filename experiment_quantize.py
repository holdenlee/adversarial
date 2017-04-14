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
from keras import backend as K

from adv_model import *
from mnist_models import *
from tf_utils import *
from train_many import *
from mix import *
from train_mix import *
from utils import *
from PIL import Image

"""
* Take normally trained net.
* Check its performance on quantized.
* Check its performance on adversarially quantized.
"""

def quantize(amt, x, bias=0):
    if amt == 0:
        return x
    return np.round((x-bias)/amt)*amt+bias

if __name__=='__main__':
    np.set_printoptions(threshold=np.nan, precision=5)
    ep=0.3
    #cnn_model_summary()
    #examine_filters(cnn_model())
    sess, models, dicts, epsilons, x, y = load_many(model1_no_dropout, filepath='pretrain5_model1_smooth0/nets', adv = fgsm2_clip, t=2)
    X_train, Y_train, X_test, Y_test = data_mnist()
    data = []
    for amt in [1, .5, .25, .125, 0]:
        X_test1 = quantize(amt, X_test)
        #np.vectorize(lambda x: quantize(x, amt), X_test)
        print("Quantization %.3f" % amt)
        data1 = []
        for (i,m) in enumerate(models):
            m.compile(optimizer='adadelta',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
            loss_and_metrics = m.evaluate(X_test1, Y_test, batch_size=128)
            print("Loss/accuracy for model %d:" % i)
            print(loss_and_metrics)
            data1.append(list(loss_and_metrics))
        data.append(data1)
    adata = []
    for amt in [1, .5, .25, .125, 0]:
        #X_test1 = quantize(amt, X_test)
        print("Quantization %.3f" % amt)
        adata1 = []
        for (i,m) in enumerate(models):
            adv_exs = sess.run(dicts[i]['adv_x'], {x:X_test, y:Y_test, epsilons[i]: ep})
            qexs = quantize(amt, adv_exs)
            acc = sess.run(dicts[i]['accuracy'], {x:qexs, y:Y_test})
            #acc = quick_avg2(sess, dicts[i]['adv_accuracy'], x, X_test1, y, Y_test, feeds = {epsilons[i]: ep})
            print("Adversarial accuracy for model %d:" % i)
            print(acc)
            adata1.append(acc)
        adata.append(adata1)
    #nonclipped training, clipped testing.
    data = np.asarray(data)
    adata = np.asarray(adata)
    print(data)
    print(adata)
    # I can't get this to work. "only length-1 arrays can be converted to Python scalars", "Mismatch between array dtype ('float64') and format specifier ('%.5f %.5f')"
    """
    with open('output/pretrain5_model1_smooth0_quantize.txt','wb') as f:
        np.savetxt(f, data, fmt='%.5f')
    with open('output/pretrain5_model1_smooth0_quantize_adv.txt','wb') as f:
        np.savetxt(f, adata, fmt='%.5f')
    """




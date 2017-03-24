from train_mix import *    
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os.path

import keras as K

from cleverhans.utils_mnist import data_mnist


def mom2(a):
    return np.dot(np.transpose(a), a)

def process_counts(a):
    #print(a.shape)
    #print(a[0:10])
    a_mom2 = mom2(a)
    (d, _) = a.shape
    a_mom2 = a_mom2/d
    #print(a_mom2)
    a_diag = np.diagonal(a_mom2)
    #print(a_diag)
    a_2 = np.outer(a_diag, a_diag)
    #print(a_2)
    a_cov = a_mom2 - a_2
    return a_mom2, a_cov

def save_corr(a, make_corr=False, write_file=None):
    if make_corr:
        a_df = pd.DataFrame(data=a) #np.transpose(a)
        a = a_df.corr()
    print(a)
    print('dim: %d' % a.shape[0])
    np.savetxt(write_file+'.txt', a)

def make_heatmap(a, make_corr=False, write_file=None):
    if make_corr:
        a_df = pd.DataFrame(data=np.transpose(a))
        a = a_df.corr()
    #ax = sns.heatmap(corrmat, square=True)
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    # Set up the matplotlib figure
    #f, ax = plt.subplots()#figsize=(12, 9))
    # Draw the heatmap using seaborn
    print(a)
    heat_map = sns.heatmap(a, square=True)
    fig = heat_map.get_figure()
    print(1)
    #plt.show()
    print(2)
    #fig = plt.figure()
    if write_file != None:
        fig.savefig(write_file+'.png')
        print(3)
        np.savetxt(write_file+'.txt', a)
        print(4)

if __name__=='__main__':
    np.set_printoptions(threshold=np.nan, precision=5)
    t=10
    dir_name = 'mix10_pretrain1_epochs100_ep0.3_reg0.5/'
    fname = 'model.ckpt-59999'
    name = os.path.join(dir_name,fname)
    epsilon = 0.3
    X_train, Y_train, X_test, Y_test = data_mnist()
    #sns.set(context="paper", font="monospace")
    K.backend.set_learning_phase(1)
    bsize = 100
    adv_model, ph_dict, epsilon1 = mix_model(t=t, many_files=False, reg_weight=0, batch_size = bsize)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, name)
    print("Model restored.")
    # batch_size * t
    #ind_correct, adv_ind_correct = sess.run([adv_model['ind_correct'], adv_model['adv_ind_correct']], feed_dict = {ph_dict['epsilon']: epsilon, ph_dict['x']: X_test, ph_dict['y'] : Y_test})
    ind_correct = []
    adv_ind_correct = []
    for i in range(10000//bsize):
        ind, adv = sess.run([adv_model['ind_correct'], adv_model['adv_ind_correct']], feed_dict = {ph_dict['epsilon']: epsilon, ph_dict['x']: X_test[bsize*i:bsize*(i+1)], ph_dict['y'] : Y_test[bsize*i:bsize*(i+1)]})
        ind_correct = ind_correct + list(ind)
        adv_ind_correct= adv_ind_correct + list(adv)
    ind_correct = np.asarray(ind_correct)
    adv_ind_correct = np.asarray(adv_ind_correct)
    ind_mom2, ind_cov = process_counts(ind_correct)
    adv_ind_mom2, adv_ind_cov = process_counts(adv_ind_correct)
    f = save_corr 
    #make_heatmap
    print('ind_cov')
    f(ind_cov, write_file=os.path.join(dir_name, 'ind_cov'))
    print('adv_ind_cov')
    f(adv_ind_cov, write_file=os.path.join(dir_name, 'adv_ind_cov'))
    print('ind_corr')
    f(ind_correct, True, write_file=os.path.join(dir_name, 'ind_corr'))
    print('adv_ind_corr')
    f(adv_ind_correct, True, write_file=os.path.join(dir_name, 'adv_ind_corr'))
    ind_correct_df = pd.DataFrame(data=ind_correct)
    

import tensorflow as tf
from utils import *

def make_adversarial_model(model, adv_f, x, *inputs):
    #(model, adv_f, epsilon, x, *inputs)
    # please feed in epsilon in the inputs
    d = model(x, *inputs)
    loss = d['loss']
    inference = d['inference']
    reg = get_with_default(d, 'regularization', 0)
    epsilon = tf.placeholder(tf.float32)
    adv_x = adv_f(x, inference, epsilon)
    adv_output = model(adv_x, *inputs)
    for s in adv_output:
        if s != 'regularization':
            d['adv_' + s] = adv_output[s]
    combined_loss = (loss + adv_output['loss']) / 2
    d['combined_loss'] = combined_loss + reg
    return d, epsilon

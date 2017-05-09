import tensorflow as tf
from utils import *
from cleverhans import utils_tf
import attacks

def fgsm2(x, predictions, eps, clip_min=None, clip_max=None):
    return attacks.fgsm2(x, predictions, eps, clip_min, clip_max)

def fgsm2_clip(x, predictions, eps):
    return attacks.fgsm2_clip(x, predictions, eps)

def fg(x, predictions):
    return attacks.fg(x,predictions)

def fgm(x, predictions, eps, clip_min=None, clip_max=None):
    return attacks.fgm(x, predictions, eps, clip_min, clip_max)

def make_adversarial_model(model, adv_f, x, *inputs):
    #(model, adv_f, epsilon, x, *inputs)
    # please feed in epsilon in the inputs
    d = model(x, *inputs)
    loss = d['loss']
    inference = d['inference']
    reg = get_with_default(d, 'regularization', 0)
    epsilon = tf.placeholder(tf.float32)
    #print("make_adversarial_model")
    #print(adv_f)
    #print(x)
    #print(epsilon)
    advs = adv_f(x,inference,epsilon)
    #print('advs:',advs)
    if isinstance(advs, tuple):
        (adv_x, adv_grad) = advs
        d['adv_grad'] = adv_grad
    else:
        adv_x=advs
    d['adv_x'] = adv_x
    adv_output = model(adv_x, *inputs)
    for s in adv_output:
        if s != 'regularization':
            d['adv_' + s] = adv_output[s]
    combined_loss = (loss + adv_output['loss']) / 2
    #combined_loss = tf.Print(combined_loss, [loss, adv_output['loss']], 'losses:')
    combined_loss = tf.identity(combined_loss, name="combined_loss")
    d['combined_loss'] = combined_loss + reg
    tf.add_to_collection('losses', d['combined_loss'])
    return d, epsilon

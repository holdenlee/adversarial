import tensorflow as tf
from utils import *
from cleverhans import utils_tf

def fgsm2(x, predictions, eps, clip_min=None, clip_max=None):
    """
    TensorFlow implementation of the Fast Gradient
    Sign method.
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param eps: the epsilon (input variation parameter)
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """
    # Define gradient of loss wrt input
    grad = fg(x, predictions)

    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    unit_grad = grad / tf.norm(grad)
    return (adv_x, unit_grad)

def fgsm2_clip(x, predictions, eps):
    return fgsm2(x, predictions, eps, 0, 1)

def fg(x, predictions):
    # Compute loss
    y = tf.to_float(
        tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True)))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    loss = utils_tf.model_loss(y, predictions, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    return grad

def fgm(x, predictions, eps, clip_min=None, clip_max=None):
    """
    TensorFlow implementation of the Fast Gradient method.
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param eps: the epsilon (input variation parameter)
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """
    grad= fg(x, predictions)
    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # NEW
    unit_grad = grad / tf.norm(grad)
    
    # Multiply by constant epsilon
    scaled_unit_grad = eps * unit_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    # Note you have to give BOTH min and max to clip. -HL
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return (adv_x, unit_grad)

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

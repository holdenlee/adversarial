import tensorflow as tf
from utils import *

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

    # Compute loss
    y = tf.to_float(
        tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True)))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    loss = utils_tf.model_loss(y, predictions, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # NEW
    unit_grad = grad / signed_grad

    # Multiply by constant epsilon
    scaled_unit_grad = eps * unit_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    # This doesn't seem right -HL
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x

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
    combined_loss = tf.identity(combined_loss, name="combined_loss")
    d['combined_loss'] = combined_loss + reg
    tf.add_to_collection('losses', d['combined_loss'])
    return d, epsilon

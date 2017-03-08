import tensorflow as tf

#solves the problem described here: http://stackoverflow.com/questions/38545362/tensorflow-variable-scope-reuse-if-variable-exists

def get_scope_variable(name, scope_name='var', shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, collections=None, caching_device=None, partitioner=None, validate_shape=True, custom_getter=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, custom_getter=custom_getter)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(name)
    return v

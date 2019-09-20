import tensorflow.compat.v1 as tf


def evaluate_tensor(func, args):
    t = func(*args)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(t)

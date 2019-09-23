import tensorflow.compat.v1 as tf


def evaluate_tensor_in_sep_process(q, func, args):
    t = func(*args)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        q.put(sess.run(t))

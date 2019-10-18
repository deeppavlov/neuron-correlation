import tensorflow.compat.v1 as tf


def evaluate_tensor_in_sep_process(q, func, args):
    t = func(*args)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        q.put(sess.run(t))


def get_counter_tensor(n):
    with tf.variable_scope('test_ds_mean_batch_indices', reuse=tf.AUTO_REUSE):
        c = tf.get_variable(
            'counter',
            shape=(),
            initializer=tf.zeros_initializer(dtype=tf.int32),
            dtype=tf.int32,
        )
        c_1d = tf.identity(tf.reshape(c, [1]))
        a = -tf.ones(shape=c_1d, dtype=tf.int32)
        b = -tf.ones(shape=n-c_1d-1, dtype=tf.int32)
        with tf.control_dependencies([c_1d]):
            increment = tf.assign_add(c, 1)
        with tf.control_dependencies([increment]):
            r = tf.identity(tf.concat([a, c_1d, b], 0))
        return r

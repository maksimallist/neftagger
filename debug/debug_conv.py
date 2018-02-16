import tensorflow as tf
import numpy as np


s_ = np.random.randn(5, 5, 6).astype(np.float32)
h_ = np.random.randn(5, 5, 4).astype(np.float32)

s = tf.placeholder(tf.float32, [None, None, 6])
h = tf.placeholder(tf.float32, [None, None, 4])

Dz = 50
winsize = 2

x = tf.layers.conv1d(tf.concat([h, s], 2), Dz, winsize*2+1, use_bias=True)
z = tf.layers.dense(x, 1, use_bias=False)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    c, c1 = sess.run([z, x], feed_dict={h: h_, s: s_})
    print(c1)

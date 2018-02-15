import tensorflow as tf
import numpy as np


batch_size = 5
L = 5
sketches_num = 5
# mass = np.zeros(L)
# input_tensor = np.random.randn(batch_size, L).astype(np.float32)
# cum_att = np.zeros((batch_size, L)).astype(np.float32)
# am = np.ones((batch_size, L))
# ones = np.ones_like(cum_att)
mass = 0.
input_tensor = np.random.randn(L).astype(np.float32)
cum_att = np.zeros(L).astype(np.float32)
am = np.ones(L)
ones = np.ones_like(cum_att)

# input = tf.placeholder(dtype=tf.float32, shape=[batch_size, L])
# b = tf.placeholder(dtype=tf.float32, shape=[batch_size, L])
# mass_ = tf.placeholder(dtype=tf.float32, shape=[batch_size])

input = tf.placeholder(dtype=tf.float32, shape=[L])
b = tf.placeholder(dtype=tf.float32, shape=[L])
mass_ = tf.constant(0.)
found_ = tf.constant(True)
# mask = [1, 1, 1, 1, 1]
mask1 = np.array([1, 1, 1, 1, 1])
mask_ = tf.placeholder(dtype=tf.int32, shape=[L], name='mask_place')
less = np.array([1, 0, 1, 1, 1])
less_ = tf.placeholder(dtype=tf.int32, shape=[L], name='mask_place')

# q_list = tf.dynamic_partition(input, mask, 2)
# condition_indices = tf.dynamic_partition(tf.range(tf.shape(input)[0]), mask, 2)  # 0 element it False,
# #  1 element if true
#
# p = q_list[1]*(1.0 - mass_)/tf.reduce_sum(q_list[1])
# p_new = tf.dynamic_stitch(condition_indices, [q_list[0], p])
#
# # verification of the condition and modification of masks
# less_mask = tf.cast(tf.less(b, p_new), tf.int32)  # 0 when u bigger than p, 1 when u less than p
# condition_indices = tf.dynamic_partition(tf.range(tf.shape(p_new)[0]), less_mask, 2)  # 0 when u bigger
# #  than p, 1 when u less than p
#
# split_p_new = tf.dynamic_partition(p_new, less_mask, 2)
# split_u = tf.dynamic_partition(b, less_mask, 2)
#
# alpha = tf.dynamic_stitch(condition_indices, [split_p_new[0], split_u[1]])
# mass_ += tf.reduce_sum(split_u[1])
#
# mask = mask * (tf.ones_like(less_mask) - less_mask)


def loop(q_, mask, mass_, found_, i, less):
    q_list = tf.dynamic_partition(q_, mask, 2)
    condition_indices = tf.dynamic_partition(tf.range(tf.shape(q_)[0]), mask, 2)  # 0 element it False,
    #  1 element if true

    p = q_list[1] * (1.0 - mass_) / tf.reduce_sum(q_list[1])
    p_new = tf.dynamic_stitch(condition_indices, [q_list[0], p])

    # verification of the condition and modification of masks
    less_mask = tf.cast(tf.less(b, p_new), tf.int32)  # 0 when u bigger than p, 1 when u less than p
    condition_indices = tf.dynamic_partition(tf.range(tf.shape(p_new)[0]), less_mask, 2)  # 0 when u bigger
    #  than p, 1 when u less than p

    split_p_new = tf.dynamic_partition(p_new, less_mask, 2)
    split_u = tf.dynamic_partition(b, less_mask, 2)

    alpha = tf.dynamic_stitch(condition_indices, [split_p_new[0], split_u[1]])
    mass_ += tf.reduce_sum(split_u[1])

    mask = mask * (tf.ones_like(less_mask) - less_mask)

    found_ = tf.cond(tf.equal(tf.reduce_sum(less_mask), 0),
                     lambda: False,
                     lambda: True)

    # if tf.reduce_sum(less_mask) < 1:
    #     found_ = False

    alpha = tf.reshape(alpha, q_.shape)

    return alpha, mask, mass_, found_, 1, less_mask


k = tf.constant(0)
(csoft, maskq, _, _, j, l_mask) = tf.while_loop(cond=lambda _0, _1, _2, f, _3, _4: f,
                                        body=loop,
                                        loop_vars=(input, mask_, mass_, found_, k, less_))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    csoft, l, _mask = sess.run([csoft, j, l_mask], feed_dict={input: am, b: ones, mass_: mass, mask_: mask1, less_: less})
    print(csoft)
    print(l)
    print(_mask)


import tensorflow as tf
import numpy as np


# def csoftmax(ten, u, temp):
#     """
#     Compute the constrained softmax (csoftmax);
#     See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
#     on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)
#
#     :param ten: input tensor
#     :param u: cumulative attention see paper
#     :param temp: softmax temperature
#     :return: distribution
#     """
#
#     shape_t = ten.shape
#     shape_u = u.shape
#     assert shape_u == shape_t
#
#     def loop(p, mask, found):
#         neg_mask = tf.ones_like(mask) - mask
#         z = tf.reduce_sum(p * mask, axis=1, keep_dims=True) / (tf.ones(shape=[shape_t[0], 1]) -
#                                                                tf.reduce_sum(neg_mask * u, axis=1,
#                                                                              keep_dims=True))
#
#         # war with NaN and inf
#         z_mask = tf.cast(tf.less_equal(z, tf.zeros_like(z)), dtype=tf.float32)
#         z = z + z_mask
#
#         alp = p / z
#
#         # verification of the condition and modification of masks
#         t_mask = tf.to_float(tf.less_equal(alp, u))
#         f_mask = tf.to_float(tf.less(u, alp))
#
#         alpha = alp * t_mask + u * f_mask
#
#         mask = mask * t_mask
#
#         if tf.reduce_sum(f_mask) == 0:
#             found = True
#
#         return alpha, mask, found
#
#     # mean
#     ten = ten - tf.reduce_mean(ten, axis=1, keep_dims=True)
#     # mask
#     mask_ = tf.ones_like(u)
#     # calculate new distribution with attention on distribution 'b'
#     q = tf.exp(ten / temp)
#     found_ = False
#
#     # start while loop
#     (q, mask_, found_) = tf.while_loop(cond=lambda _0, _1, f: f is False,
#                                        body=loop,
#                                        loop_vars=(q, mask_, found_))
#
#     return q


# def csoftmax(ten, u, temp=1.0):
#     """
#     Compute the constrained softmax (csoftmax);
#     See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
#     on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)
#
#     :param ten: input tensor
#     :param u: cumulative attention see paper
#     :param temp: softmax temperature
#     :return: distribution
#     """
#
#     shape_t = ten.shape
#     shape_u = u.shape
#     assert shape_u == shape_t
#
#     def loop(p, mask, i, found):
#         neg_mask = tf.ones_like(mask) - mask
#         p_ = p * neg_mask
#
#         z = tf.reduce_sum(p * mask, axis=1, keep_dims=True) / (tf.ones(shape=[shape_t[0], 1]) -
#                                                                tf.reduce_sum(neg_mask * u, axis=1, keep_dims=True))
#         # war with NaN and inf
#         z_mask = tf.cast(tf.less_equal(z, tf.zeros_like(z)), dtype=tf.float32)
#         z = z + z_mask
#
#         alp = (p * mask)/z
#         alp = alp + p_
#
#         # verification of the condition and modification of masks
#         t_mask = tf.to_float(tf.less_equal(alp, u))
#         f_mask = tf.to_float(tf.less(u, alp))
#
#         alpha = alp * t_mask + u * f_mask
#
#         mask = mask * t_mask
#
#         if tf.reduce_mean(t_mask) == 1:
#             found = True
#
#         return alpha, mask, i+1, found
#
#     # mean
#     ten = ten - tf.reduce_mean(ten, axis=1, keep_dims=True)
#     # mask
#     mask_ = tf.ones_like(u)
#     # mask_ = tf.zeros_like(u)
#     # calculate new distribution with attention on distribution 'b'
#     q = tf.exp(ten / temp)
#     found_ = False
#
#     # start while loop
#     k = 0
#     (r, _, l, _) = tf.while_loop(cond=lambda _0, _1, _2, f: f is False,
#                                  body=loop,
#                                  loop_vars=(q, mask_, k, found_))
#
#     return r, l


def constrained_softmax(z, u):
    z -= np.mean(z)
    q = np.exp(z)
    active = np.ones(len(u))
    mass = 0.
    p = np.zeros(len(z))
    while True:
        inds = active.nonzero()[0]
        p[inds] = q[inds] * (1. - mass) / np.sum(q[inds])
        found = False
        for i in inds:
            if p[i] > u[i]:
                p[i] = u[i]
                mass += u[i]
                found = True
                active[i] = 0
        if not found:
            break

    return p, active, mass


def csoftmax(tensor, cumulative_att, t):

    def csoftmax_for_slice(input):
        """
        Compute the constrained softmax (csoftmax);
        See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
        on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)

        :param input: [input tensor, cumulative attention]
        :return: distribution
        """

        [ten, u] = input

        shape_t = ten.shape
        shape_u = u.shape
        assert shape_u == shape_t

        ten -= tf.reduce_mean(ten)
        q = tf.exp(ten)
        active = tf.ones_like(u, dtype=tf.int32)
        mass = tf.constant(0, dtype=tf.float32)
        found = tf.constant(True, dtype=tf.bool)

        def loop(q_, mask, mass_, found_):
            q_list = tf.dynamic_partition(q_, mask, 2)
            condition_indices = tf.dynamic_partition(tf.range(tf.shape(q_)[0]), mask, 2)  # 0 element it False,
            #  1 element if true

            p = q_list[1]*(1.0 - mass_)/tf.reduce_sum(q_list[1])
            p_new = tf.dynamic_stitch(condition_indices, [q_list[0], p])

            # verification of the condition and modification of masks
            less_mask = tf.cast(tf.less(u, p_new), tf.int32)  # 0 when u bigger than p, 1 when u less than p
            condition_indices = tf.dynamic_partition(tf.range(tf.shape(p_new)[0]), less_mask, 2)  # 0 when u bigger
            #  than p, 1 when u less than p

            split_p_new = tf.dynamic_partition(p_new, less_mask, 2)
            split_u = tf.dynamic_partition(u, less_mask, 2)

            alpha = tf.dynamic_stitch(condition_indices, [split_p_new[0], split_u[1]])
            mass_ += tf.reduce_sum(split_u[1])

            mask = mask * (tf.ones_like(less_mask) - less_mask)

            found_ = tf.cond(tf.equal(tf.reduce_sum(less_mask), 0),
                             lambda: False,
                             lambda: True)

            alpha = tf.reshape(alpha, q_.shape)

            return alpha, mask, mass_, found_

        (csoft, mask_, _, _) = tf.while_loop(cond=lambda _0, _1, _2, f: f,
                                             body=loop,
                                             loop_vars=(q, active, mass, found))

        return [csoft, mask_]

    shape_ten = tensor.shape
    shape_cum = cumulative_att.shape
    assert shape_cum == shape_ten

    t_in = [tensor, cumulative_att]
    cs, _ = tf.map_fn(csoftmax_for_slice, t_in, dtype=[tf.float32, tf.float32])  # [bs, L]
    return cs


################################################################

################################################################
state_size = 40
dim_hlayer = 20
window_size = 2
batch_size = 5
activation = tf.nn.tanh
L = 5
sketches_num = 5
discount_factor = 0
temperature = 1.

input = tf.placeholder(dtype=tf.float32, shape=[batch_size, L])
b = tf.placeholder(dtype=tf.float32, shape=[batch_size, L])

input_tensor = np.random.randn(batch_size, L).astype(np.float32)
cum_att = np.zeros((batch_size, L)).astype(np.float32)
print('input random tensor: \n{}'.format(input_tensor))
print('input zeros cumulative attention: \n{}'.format(cum_att))

active_mask = tf.placeholder(dtype=tf.float32, shape=[batch_size, L])

# tens = csoftmax(input, b, active_mask)
tens = csoftmax(input, b, temperature)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # t = sess.run(tens, feed_dict={input: input_tensor, b: cum_att})

    am = np.ones((batch_size, L))
    ones = np.ones_like(cum_att)

    print('iterations: \n{}'.format(sketches_num), '\n')
    for i in range(sketches_num):
        print('input attention: \n{}\n'.format(cum_att), '\n')
        # t = sess.run(tens, feed_dict={input: input_tensor, b: ones - cum_att, active_mask: am})
        t = sess.run(tens, feed_dict={input: input_tensor, b: ones - cum_att})
        cum_att += t
        print('Iteration: {}'.format(i+1))
        # print('Iteration in loop: {}'.format(j))
        print('csoftmax: \n{}\n'.format(t), '\n')
        print('csoftmax sum: \n{}\n'.format(np.sum(t, axis=1)), '\n')
        # print('input tensor: \n{}\n'.format(t['input_tensor']), '\n')

        # print('input active mask: \n{}\n'.format(t['input_active_mask']), '\n')
        # print('input non active mask: \n{}\n'.format(t['input non active mask']), '\n')
        # print('input tensor after mean: \n{}\n'.format(t['tensor']), '\n')
        #
        # print('Q: \n{}\n'.format(t['Q']), '\n')
        # print('Z: \n{}\n'.format(t['Z']), '\n')
        # print('A: \n{}\n'.format(t['A']), '\n')
        # print('alpha: \n{}\n'.format(t['alpha']), '\n')
        #
        # print('true less mask: \n{}\n'.format(t['t_mask']), '\n')
        # print('false less mask: \n{}\n'.format(t['f_mask']), '\n')
        # print('new active mask: \n{}\n'.format(t['active_mask']), '\n')


tf.reset_default_graph()

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


def csoftmax(ten, u, temp=1.0):
    """
    Compute the constrained softmax (csoftmax);
    See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
    on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)

    :param ten: input tensor
    :param u: cumulative attention see paper
    :param temp: softmax temperature
    :return: distribution
    """

    shape_t = ten.shape
    shape_u = u.shape
    assert shape_u == shape_t

    def loop(p, mask, i, found):
        neg_mask = tf.ones_like(mask) - mask

        z = tf.reduce_sum(p * mask, axis=1, keep_dims=True) / (tf.ones(shape=[shape_t[0], 1]) -
                                                               tf.reduce_sum(neg_mask * u, axis=1, keep_dims=True))
        # war with NaN and inf
        z_mask = tf.cast(tf.less_equal(z, tf.zeros_like(z)), dtype=tf.float32)
        z = z + z_mask

        alp = (p * mask)/z

        # verification of the condition and modification of masks
        t_mask = tf.to_float(tf.less_equal(alp, u))
        f_mask = tf.to_float(tf.less(u, alp))

        alpha = alp * t_mask + u * f_mask

        mask = mask * t_mask

        if tf.reduce_sum(f_mask) == 0:
            found = True

        return alpha, mask, i+1, found

    # mean
    ten = ten - tf.reduce_mean(ten, axis=1, keep_dims=True)
    # mask
    mask_ = tf.ones_like(u)
    # mask_ = tf.zeros_like(u)
    # calculate new distribution with attention on distribution 'b'
    q = tf.exp(ten / temp)
    found_ = False

    # start while loop
    k = 0
    (r, _, l, _) = tf.while_loop(cond=lambda _0, _1, _2, f: f is False,
                                 body=loop,
                                 loop_vars=(q, mask_, k, found_))

    return r, l


def my_csoftmax(ten, u, mask, temp):
    """
    Compute the constrained softmax (csoftmax);
    See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
    on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)

    :param ten: input tensor
    :param u: cumulative attention see paper
    :param mask: mask with active elements
    :param temp: softmax temperature
    :return: distribution
    """

    shape_t = ten.shape
    shape_u = u.shape
    assert shape_u == shape_t

    # mean
    ten = ten - tf.reduce_mean(ten, axis=1, keep_dims=True)

    neg_mask = tf.ones_like(mask) - mask

    # calculate new distribution with attention on distribution 'b'
    Q = tf.exp(ten/temp)

    # TODO: it is really need ? we wanted some corelation with Q
    # u = u/temp

    Z = tf.reduce_sum(Q*mask, axis=1, keep_dims=True)/(tf.ones(shape=[shape_t[0], 1]) -
                                                       tf.reduce_sum(neg_mask*u, axis=1, keep_dims=True))

    # war with NaN and inf
    z_mask = tf.cast(tf.less_equal(Z, tf.zeros_like(Z)), dtype=tf.float32)
    Z = Z + z_mask

    A = Q / Z

    # verification of the condition and modification of masks
    t_mask = tf.to_float(tf.less_equal(A, u))
    f_mask = tf.to_float(tf.less(u, A))

    alpha = A * t_mask + u * f_mask

    mask = mask * t_mask

    return alpha, mask


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
        t, j = sess.run(tens, feed_dict={input: input_tensor, b: ones - cum_att})
        cum_att += t
        print('Iteration: {}'.format(i+1))
        print('Iteration in loop: {}'.format(j))
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

import tensorflow as tf
import numpy as np


def csoftmax_paper(tensor, u, mask):
    """
    Compute the constrained softmax (csoftmax);
    See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
    on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)

    :param tensor: input tensor
    :param u: cumulative attention see paper
    :param mask: mask with active elements
    :return: distribution
    """

    shape_t = tensor.shape
    shape_b = b.shape
    assert shape_b == shape_t

    # mean
    tensor = tensor - tf.reduce_mean(tensor, axis=1, keep_dims=True)

    neg_mask = tf.ones_like(mask) - mask

    # calculate new distribution with attention on distribution 'b'
    Q = tf.exp(tensor)
    Z = tf.reduce_sum(Q*mask, axis=1, keep_dims=True) / (tf.ones(shape=[shape_t[0], 1]) - tf.reduce_sum(neg_mask*u,
                                                                                                        axis=1,
                                                                                                        keep_dims=True))

    # war with NaN and inf
    z_mask = tf.cast(tf.less_equal(Z, tf.zeros_like(Z)), dtype=tf.float32)
    Z = Z + z_mask

    A = Q/Z

    # verification of the condition and modification of masks
    t_mask = tf.to_float(tf.less_equal(A, u))
    f_mask = tf.to_float(tf.less(u, A))

    alpha = A * t_mask + u * f_mask

    mask = mask * t_mask

    return alpha, mask


def csoftmax_paper_test(tensor, u, mask):

    tensors = dict()
    tensors['input_tensor'] = tensor
    tensors['input_active_mask'] = mask
    tensors['input_attention'] = u

    shape_t = input_tensor.shape
    shape_b = b.shape
    assert shape_b == shape_t

    # mean
    tensor = tensor - tf.reduce_mean(input_tensor, axis=1, keep_dims=True)
    tensors['tensor'] = tensor

    neg_mask = tf.ones_like(mask) - mask
    tensors['input non active mask'] = neg_mask

    # calculate new distribution with attention on distribution 'b'
    Q = tf.exp(tensor)
    Z = tf.reduce_sum(Q * mask, axis=1, keep_dims=True)/(tf.ones(shape=[shape_t[0], 1]) - tf.reduce_sum(neg_mask*u,
                                                                                                        axis=1,
                                                                                                        keep_dims=True))

    # war with NaN and inf
    z_mask = tf.cast(tf.less_equal(Z, tf.zeros_like(Z)), dtype=tf.float32)
    Z = Z + z_mask

    A = Q / Z

    # verification of the condition and modification of masks
    t_mask = tf.to_float(tf.less_equal(A, u))
    f_mask = tf.to_float(tf.less(u, A))

    alpha = A * t_mask + u * f_mask
    tensors['alpha'] = alpha

    mask = mask * t_mask

    tensors['Q'] = Q
    tensors['Z'] = Z
    tensors['A'] = A
    tensors['csoftmax'] = alpha
    tensors['t_mask'] = t_mask
    tensors['f_mask'] = f_mask
    tensors['active_mask'] = mask

    return tensors


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
print('random tensor: \n{}'.format(input_tensor))
print(cum_att)

active_mask = tf.placeholder(dtype=tf.float32, shape=[batch_size, L])

tens = csoftmax_paper_test(input, b, active_mask)

# def iter(t, b):
#
#
#
#     tensor_list = []
#     k = 0
#
#     # for i in range(5):
#     #     ten = csoftmax_paper(t, b, active_i, not_active_i)
#     #     tensor_list.append(ten)
#     #
#     #     b += ten['csoftmax']
#     #     active_i = ten['active_mask']
#     #     not_active_i = ten['non_active_mask']
#     #     k += 1
#     #
#     #     if not_active_i == 1:
#     #         break
#
#     ten = csoftmax_paper(t, b, active_i, not_active_i)
#     tensor_list.append(ten)
#
#     b += ten['csoftmax']
#     active_i = ten['active_mask']
#     not_active_i = ten['non_active_mask']
#
#     return tensor_list, i
#
#
# ten, k = iter(input, b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # t = sess.run(tens, feed_dict={input: input_tensor, b: cum_att})

    p = 5

    am = np.ones((batch_size, L))
    ones = np.ones_like(cum_att)

    print('iterations: \n{}'.format(p), '\n')
    for i in range(p):
        t = sess.run(tens, feed_dict={input: input_tensor, b: ones - cum_att, active_mask: am})
        cum_att += t['csoftmax']
        am = t['active_mask']
        print('Iteration: {}'.format(i+1))
        # if np.mean(am) == 0:
        #     break

    # print('input tensor: \n{}\n'.format(t['input_tensor'][i]), '\n')
    # print('input cumulative attention: \n{}\n'.format(t['b'][i]), '\n')
    # print('inverse cumulative attention: \n{}\n'.format(t['u'][i]), '\n')
    # print('input active mask: \n{}\n'.format(t['input_active_mask'][i]), '\n')
    # print('input non active mask: \n{}\n'.format(t['input_non_active_mask'][i]), '\n')
    # print('input tensor after mean: \n{}\n'.format(t['tensor'][i]), '\n')
    #
    # print('a: \n{}\n'.format(t['a'][i]), '\n')
    # print('f: \n{}\n'.format(t['f'][i]), '\n')
    # print('Z: \n{}\n'.format(t['z'][i]), '\n')
    # print('alpha: \n{}\n'.format(t['alpha'][i]), '\n')
    #
    # print('true less mask: \n{}\n'.format(t['t_mask'][i]), '\n')
    # print('false less mask: \n{}\n'.format(t['f_mask'][i]), '\n')
    # print('new active mask: \n{}\n'.format(t['active_mask'][i]), '\n')
    # print('new non active mask: \n{}\n'.format(t['non_active_mask'][i]), '\n')
    # print('csoftmax: \n{}\n'.format(t['csoftmax'][i]), '\n')

        print('input tensor: \n{}\n'.format(t['input_tensor']), '\n')
        print('input attention: \n{}\n'.format(t['input_attention']), '\n')
        print('input active mask: \n{}\n'.format(t['input_active_mask']), '\n')
        print('input non active mask: \n{}\n'.format(t['input non active mask']), '\n')
        print('input tensor after mean: \n{}\n'.format(t['tensor']), '\n')

        print('Q: \n{}\n'.format(t['Q']), '\n')
        print('Z: \n{}\n'.format(t['Z']), '\n')
        print('A: \n{}\n'.format(t['A']), '\n')
        print('alpha: \n{}\n'.format(t['alpha']), '\n')

        print('true less mask: \n{}\n'.format(t['t_mask']), '\n')
        print('false less mask: \n{}\n'.format(t['f_mask']), '\n')
        print('new active mask: \n{}\n'.format(t['active_mask']), '\n')
        print('csoftmax: \n{}\n'.format(t['csoftmax']), '\n')

tf.reset_default_graph()

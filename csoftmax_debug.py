import tensorflow as tf
import numpy as np


def constrained_softmax(input_tensor, b, temp):
    """
    Compute the constrained softmax (csoftmax);
    See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
    on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)

    :param input_tensor: input tensor
    :param b: cumulative attention see paper
    :param temp: softmax temperature
    :return: distribution
    """

    # input_tensor = tf.reduce_mean(input_tensor)

    row_max = tf.expand_dims(tf.reduce_max(input_tensor, 1), 1)
    input_tensor = input_tensor - row_max

    z = tf.reduce_sum(tf.exp(input_tensor / temp), axis=1, keep_dims=True)
    a = tf.exp(input_tensor / temp) * (b / temp) / z
    # a = tf.exp(input_tensor/temp) * b / z
    u = tf.ones_like(b) - b
    t_mask = tf.to_float(tf.less_equal(a, u))
    f_mask = tf.to_float(tf.less(u, a))
    A = a * t_mask
    U = u * f_mask

    csoftmax = A + U

    tensors = dict()
    tensors['csoftmax'] = csoftmax
    tensors['t_mask'] = t_mask
    tensors['f_mask'] = f_mask
    tensors['A'] = A
    tensors['U'] = U
    tensors['a'] = a
    tensors['u'] = u
    tensors['z'] = z
    tensors['b'] = b
    tensors['input_tensor'] = input_tensor

    return tensors


def csoftmax_(input_tensor, b, temp):
    """
    Compute the constrained softmax (csoftmax);
    See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
    on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)

    :param input_tensor: input tensor
    :param b: cumulative attention see paper
    :param temp: softmax temperature
    :return: distribution
    """

    # Z = tf.reduce_mean(tensor, axis=1)
    # s = tf.zeros([shape_t[0]])

    shape_t = input_tensor.shape
    shape_b = b.shape
    assert shape_b == shape_t

    # initialize
    tensor = input_tensor - tf.reduce_mean(input_tensor, axis=1)

    active = tf.ones_like(tensor)
    non_active = (tf.ones_like(tensor) - active)
    ones = tf.ones([shape_t[0]])
    q = tf.exp(tensor)

    u = tf.ones_like(b) - b

    for i in range(shape_t[1]):
        alpha = (q*active/temp)*(ones - tf.reduce_mean(u*non_active/temp, axis=1))/(tf.reduce_mean(q*active, axis=1))

        t_mask = tf.to_float(tf.less_equal(alpha, u))
        f_mask = tf.to_float(tf.less(u, alpha))

        active = active*f_mask
        non_active = non_active*t_mask
        q = alpha*t_mask + u*f_mask

    # tensors = dict()
    # tensors['csoftmax'] = csoftmax
    # tensors['t_mask'] = t_mask
    # tensors['f_mask'] = f_mask
    # tensors['A'] = A
    # tensors['U'] = U
    # tensors['a'] = a
    # tensors['u'] = u
    # tensors['z'] = z
    # tensors['b'] = b
    # tensors['input_tensor'] = input_tensor

    return alpha


def csoftmax_paper(input_tensor, b, active, non_active):
    """
    Compute the constrained softmax (csoftmax);
    See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
    on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)

    :param input_tensor: input tensor
    :param b: cumulative attention see paper
    :param temp: softmax temperature
    :return: distribution
    """
    tensors = dict()
    tensors['input_tensor'] = input_tensor
    tensors['input_active_mask'] = active
    tensors['input_non_active_mask'] = non_active

    shape_t = input_tensor.shape
    shape_b = b.shape
    assert shape_b == shape_t

    # mean
    tensor = input_tensor - tf.reduce_mean(input_tensor, axis=1)
    tensors['tensor'] = tensor
    #
    ones = tf.ones([shape_t[0]])
    q = tf.exp(tensor)
    u = tf.ones_like(b) - b

    # calculate new distribution with attention on distribution 'b'
    a = q*active
    z = tf.reduce_mean(q*active, axis=1)
    f = ones - tf.reduce_mean(u*non_active, axis=1)

    z_mask = tf.cast(tf.less_equal(z, tf.zeros_like(z)), dtype=tf.float32)
    z = z + z_mask

    alpha = a*f/z
    tensors['alpha'] = alpha

    # verification of the condition and modification of masks
    t_mask = tf.to_float(tf.less_equal(alpha, u))
    f_mask = tf.to_float(tf.less(u, alpha))

    alpha = alpha * t_mask + u * f_mask

    active = active - f_mask
    non_active = non_active + f_mask

    tensors['a'] = a
    tensors['u'] = u
    tensors['z'] = z
    tensors['b'] = b
    tensors['f'] = f
    tensors['csoftmax'] = alpha
    tensors['t_mask'] = t_mask
    tensors['f_mask'] = f_mask
    tensors['active_mask'] = active
    tensors['non_active_mask'] = non_active

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
non_active_mask = tf.placeholder(dtype=tf.float32, shape=[batch_size, L])

tens = csoftmax_paper(input, b, active_mask, non_active_mask)

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
    nam = np.zeros((batch_size, L))

    print('iterations: \n{}'.format(p), '\n')
    for i in range(p):
        print('input cumulative attention: \n{}\n'.format(cum_att), '\n')
        t = sess.run(tens, feed_dict={input: input_tensor, b: cum_att, active_mask: am, non_active_mask: nam})
        cum_att += t['csoftmax']
        am = t['active_mask']
        nam = t['non_active_mask']
        print('Iteration: {}'.format(i+1))
        if np.mean(am) == 0:
            break

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

        print('inverse cumulative attention: \n{}\n'.format(t['u']), '\n')
        print('input active mask: \n{}\n'.format(t['input_active_mask']), '\n')
        print('input non active mask: \n{}\n'.format(t['input_non_active_mask']), '\n')
        print('input tensor after mean: \n{}\n'.format(t['tensor']), '\n')

        print('a: \n{}\n'.format(t['a']), '\n')
        print('f: \n{}\n'.format(t['f']), '\n')
        print('Z: \n{}\n'.format(t['z']), '\n')
        print('alpha: \n{}\n'.format(t['alpha']), '\n')

        print('true less mask: \n{}\n'.format(t['t_mask']), '\n')
        print('false less mask: \n{}\n'.format(t['f_mask']), '\n')
        print('new active mask: \n{}\n'.format(t['active_mask']), '\n')
        print('new non active mask: \n{}\n'.format(t['non_active_mask']), '\n')
        print('csoftmax: \n{}\n'.format(t['csoftmax']), '\n')

tf.reset_default_graph()

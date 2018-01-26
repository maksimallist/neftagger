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


def csoftmax_paper(input_tensor, b, temp):
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
# print(cum_att)

constrained_weights = csoftmax_paper(input, b, temperature)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    results = sess.run(constrained_weights, feed_dict={input: input_tensor, b: cum_att})

    print('csoftmax: \n{}'.format(results), '\n')

    # print('input cumulative attention: {}\n'.format(results['b']), '\n')
    # print('inverse cumulative attention: {}\n'.format(results['u']), '\n')
    # print('Z: {}\n'.format(results['z']), '\n')
    # print('a: {}\n'.format(results['a']), '\n')
    # print('true less mask: {}\n'.format(results['t_mask']), '\n')
    # print('false less mask: {}\n'.format(results['f_mask']), '\n')
    # print('A: {}\n'.format(results['A']), '\n')
    # print('U: {}\n'.format(results['U']), '\n')
    # print('csoftmax: {}\n'.format(results['csoftmax']), '\n')

tf.reset_default_graph()

import numpy as np
import pdb

batch_size = 1
L = 3
sketches_num = 5
discount_factor = 0
temperature = 1.

input_tensor = np.random.randn(L).astype(np.float32)
cum_att = np.zeros(L).astype(np.float32)
my_input_tensor = np.random.randn(batch_size, L).astype(np.float32)
my_cum_att = np.zeros((batch_size, L)).astype(np.float32)
print('initialize random tensor: \n{}'.format(input_tensor))
print('initialize zeros cumulative attention: \n{}'.format(cum_att))


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


# realization with z.shape = u.shape = [bs, L]
def my_constrained_softmax(z, u):
    # initialization
    z -= np.mean(z, axis=1)
    q = np.exp(z)
    active = np.ones_like(z)

    # my mask
    non_active = np.ones_like(z) - active
    # need ?
    p = np.zeros_like(z)

    while True:
        # found = False

        # inds = active.nonzero()[0]
        p_ = p * non_active
        z = np.sum(q*active, axis=1) / (np.ones(u.shape[0]) - np.sum(u*non_active, axis=1))

        # war with NaN and inf
        z_mask = np.less_equal(z, np.zeros_like(z)).astype(np.float32)
        z = z + z_mask

        p = (q*active) / z
        p = p + p_

        # verification of the condition and modification of masks
        t_mask = np.less_equal(p, u).astype(np.float32)
        f_mask = np.less(u, p).astype(np.float32)

        p = p * t_mask + u * f_mask

        active = active * t_mask

        if np.mean(t_mask) == 1:
            break

    return p, active


alpha, mask, mass = constrained_softmax(input_tensor, np.ones_like(cum_att) - cum_att)
my_alpha, my_mask = my_constrained_softmax(my_input_tensor, np.ones_like(my_cum_att) - my_cum_att)

pdb.set_trace()

# for i in range(sketches_num):
#     print('\nIteration: {}'.format(i+1))
#     alpha, mask, mass = constrained_softmax(input_tensor, np.ones_like(cum_att) - cum_att)
#     print('Standart constrained softmax: {}'.format(alpha))
#     print('Standart mask: {}'.format(mask))
#     print('Standart mass: {}'.format(mass))
#     print('Sum of alpha: {}'.format(sum(alpha)))
#     cum_att += alpha
#
#     my_alpha, my_mask = my_constrained_softmax(my_input_tensor, np.ones_like(my_cum_att) - my_cum_att)
#     print('\nMy constrained softmax: {}'.format(my_alpha))
#     print('My mask: {}'.format(my_mask))
#     print('Sum of my alpha: {}'.format(np.sum(my_alpha, axis=1)))
#     my_cum_att += my_alpha

# # standart
# print('Standart ...')
# my_input_tensor -= np.mean(my_input_tensor, axis=1)
# softmax = np.exp(my_input_tensor)/np.sum(np.exp(my_input_tensor), axis=1)
# print('exp input: {}'.format(np.exp(my_input_tensor)))
# print('znam: {}'.format(np.sum(my_input_tensor, axis=1)))
# print('softmax: {}\n'.format(softmax))
#
# # my
# print('My ...')
# my_cum_att = np.ones_like(my_cum_att) - my_cum_att
# q = np.exp(my_input_tensor)
# print('exp input: {}'.format(q))
# active = np.ones_like(my_input_tensor)
# print('active: {}'.format(active))
# # my mask
# non_active = np.ones_like(my_input_tensor) - active
# print('non_active: {}'.format(non_active))
#
# p = np.zeros_like(my_input_tensor)
# print('p: {}'.format(p))
#
# p_ = p * non_active
# print('p_ inverse: {}'.format(p_))
#
# A = q * active
# print('A', A)
# B = my_cum_att * non_active
# print('B', B)
# A_sum = np.sum(A, axis=1)
# print('A_sum', A_sum, A_sum.shape)
# B_sum = np.sum(B, axis=1)
# print('B_sum', B_sum)
# C = np.ones(my_cum_att.shape[0]) - B_sum
# print('C', C, C.shape)
# z = A_sum / C
# print('Z: {} {}'.format(z, z.shape))
#
# # war with NaN and inf
# z_mask = np.less_equal(z, np.zeros_like(z)).astype(np.float32)
# print('z_mask: {}'.format(z_mask))
# z = z + z_mask
# print('Z after mask: {}'.format(z))
# z = np.reshape(z, (3, 1))
# p = (q * active) / z
# print('alpha: {} {}'.format(p, p.shape))
# p = p + p_
# print('sum of p: {}'.format(p))
# # verification of the condition and modification of masks
# t_mask = np.less_equal(p, my_cum_att).astype(np.float32)
# print('t_mask: {}'.format(t_mask))
# f_mask = np.less(my_cum_att, p).astype(np.float32)
# print('f_mask: {}'.format(f_mask))
# p = p * t_mask + my_cum_att * f_mask
# print('csoftmax: {}'.format(p))
# active = active * t_mask
# print('active mask itog: {}'.format(active))
#
#
# print('standart softmax: {}'.format(softmax))
# print('my softmax: {}'.format(p))

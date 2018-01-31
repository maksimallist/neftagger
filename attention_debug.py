import numpy as np
import tensorflow as tf


# # Attention block
# def attention_block(hidden_states, state_size, window_size, dim_hlayer, batch_size,
#                     activation, L, sketches_num, discount_factor, temperature):
#
#     with tf.variable_scope('loop_matrices', reuse=tf.AUTO_REUSE):
#         W_hsz = tf.get_variable(name="W_hsz", shape=[2 * state_size * (2 * window_size + 1), dim_hlayer],
#                                 initializer=tf.contrib.layers.xavier_initializer(uniform=True,
#                                                                                  dtype=tf.float32))
#
#         w_z = tf.get_variable(name="w_z", shape=[dim_hlayer],
#                               initializer=tf.random_uniform_initializer(dtype=tf.float32))
#
#         v = tf.get_variable(name="v", shape=[dim_hlayer, 1],
#                             initializer=tf.random_uniform_initializer(dtype=tf.float32))
#
#         W_hh = tf.get_variable(name="W_hh", shape=[2 * state_size * (2 * window_size + 1), state_size],
#                                initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
#
#         w_h = tf.get_variable(name="w_h", shape=[state_size],
#                               initializer=tf.random_uniform_initializer(dtype=tf.float32))
#
#     def conv_r(padded_matrix, r):
#         """
#         Extract r context columns around each column and concatenate
#         :param padded_matrix: batch_size x L+(2*r) x 2*state_size
#         :param r: context size
#         :return:
#         """
#         # gather indices of padded
#         time_major_matrix = tf.transpose(padded_matrix,
#                                          [1, 2, 0])  # time-major  -> L x 2*state_size x batch_size
#         contexts = []
#         for j in np.arange(r, L + r):
#             # extract 2r+1 rows around i for each batch
#             context_j = time_major_matrix[j - r:j + r + 1, :, :]  # 2*r+1 x 2*state_size x batch_size
#             # concatenate
#             context_j = tf.reshape(context_j,
#                                    [(2 * r + 1) * 2 * state_size, batch_size])  # (2*r+1)*(state_size) x batch_size
#             contexts.append(context_j)
#         contexts = tf.stack(contexts)  # L x (2*r+1)* 2*(state_size) x batch_size
#         batch_major_contexts = tf.transpose(contexts, [2, 0, 1])
#         # switch back: batch_size x L x (2*r+1)*2(state_size) (batch-major)
#         return batch_major_contexts
#
#     def prepare_tensor(hidstates, sk, padding_col):
#         hs = tf.concat([hidstates, sk], 2)
#         # add column on right and left, and add context window
#         hs = tf.pad(hs, padding_col, "CONSTANT", name="HS_padded")
#         hs = conv_r(hs, window_size)  # [batch_size, L, 2*state*(2*window_size + 1)]
#         return hs
#
#     def sketch_step(tensor, cum_attention, mask, neg_mask, temper):
#
#         def csoftmax(input_tensor, b, active, non_active, temp):
#             """
#             Compute the constrained softmax (csoftmax);
#             See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
#             on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)
#
#             :param input_tensor: input tensor
#             :param b: cumulative attention see paper
#             :param temp: softmax temperature
#             :return: distribution
#             """
#
#             shape_t = input_tensor.shape
#             shape_b = b.shape
#             assert shape_b == shape_t
#
#             # mean
#             # tensor = input_tensor - tf.reduce_mean(input_tensor, axis=1)
#             tensor = input_tensor
#             #
#             ones = tf.ones([shape_t[0]])
#             q = tf.exp(tensor)
#             u = tf.ones_like(b) - b
#
#             # calculate new distribution with attention on distribution 'b'
#             A = (q * active / temp)
#             C = (ones - tf.reduce_mean(u * non_active / temp, axis=1))
#             Z = (tf.reduce_mean(q * active, axis=1))
#
#             alpha = A * tf.reshape(C, [shape_t[0], 1]) / tf.reshape(Z, [shape_t[0], 1])
#
#             # verification of the condition and modification of masks
#             t_mask = tf.to_float(tf.less_equal(alpha, u))
#             f_mask = tf.to_float(tf.less(u, alpha))
#
#             alpha = alpha * t_mask + u * f_mask
#
#             active = active - f_mask
#             non_active = non_active + f_mask
#
#             return alpha, active, non_active
#
#         def attention(t):
#             before_att = activation(tf.matmul(t, W_hsz) + w_z)
#             att = tf.matmul(before_att, v)  # [batch_size, 1]
#             return att
#
#         tensor = tf.transpose(tensor, [1, 0, 2])
#
#         attentions = tf.map_fn(attention, tensor, dtype=tf.float32)  # [batch_size, 1, L]
#         attentions = tf.reshape(attentions, [batch_size, L]) - cum_attention*discount_factor  # [batch_size, L]
#         constrained_weights, new_mask, new_neg_mask = csoftmax(attentions, cum_attention, mask, neg_mask, temper)  # [batch_size, L]
#
#         tensor = tf.transpose(tensor, [1, 0, 2])
#         cn = tf.reduce_sum(tensor*tf.expand_dims(constrained_weights, [2]), axis=1)  # [batch_size,
#         #  2*state_size*(2*window_size + 1)]
#         cn = tf.reshape(cn, [batch_size, 2*state_size*(2*window_size + 1)])  # [batch_size,
#         #  2*state_size*(2*window_size + 1)]
#         s = activation(tf.matmul(cn, W_hh) + w_h)  # [batch_size, state_size]
#
#         s = tf.matmul(tf.expand_dims(constrained_weights, [2]), tf.expand_dims(s, [1]))  # [batch_size, L,
#         #  state_size]
#
#         return s, constrained_weights, new_mask, new_neg_mask
#
#     sketch = tf.zeros(shape=[batch_size, L, state_size], dtype=tf.float32)  # sketch tenzor
#     cum_att = tf.zeros(shape=[batch_size, L])  # cumulative attention
#     padding_hs_col = tf.constant([[0, 0], [window_size, window_size], [0, 0]], name="padding_hs_col")
#     temperature = tf.constant(temperature, dtype=tf.float32, name='attention_temperature')
#
#     mask_i = tf.ones([batch_size, L])
#     neg_mask_i = (tf.ones_like(mask_i) - mask_i)
#
#     sketches = []
#     cum_attentions = []
#
#     for i in range(sketches_num):
#         sketch_, cum_att_, mask_i, neg_mask_i = sketch_step(prepare_tensor(hidden_states, sketch, padding_hs_col),
#                                                             cum_att, mask_i, neg_mask_i, temperature)
#         # print(cum_att_.eval(session=sess))
#         sketch += sketch_
#         cum_att += cum_att_
#         sketches.append(sketch_)  # list of tensors with shape [batch_size, L, state_size]
#         cum_attentions.append(cum_att_)  # list of tensors with shape [batch_size, L]
#
#         if tf.reduce_sum(mask_i) == 0:
#             break
#
#     return sketches, cum_attentions


# Attention block
def attention_block(hidden_states, state_size, window_size, dim_hlayer, batch_size,
                    activation, L, sketches_num, discount_factor, temperature, full_model):

    with tf.variable_scope('loop_matrices', reuse=tf.AUTO_REUSE):
        W_hsz = tf.get_variable(name="W_hsz", shape=[2 * state_size * (2 * window_size + 1), dim_hlayer],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True,
                                                                                 dtype=tf.float32))

        w_z = tf.get_variable(name="w_z", shape=[dim_hlayer],
                              initializer=tf.random_uniform_initializer(dtype=tf.float32))

        v = tf.get_variable(name="v", shape=[dim_hlayer, 1],
                            initializer=tf.random_uniform_initializer(dtype=tf.float32))

        W_hh = tf.get_variable(name="W_hh", shape=[2 * state_size * (2 * window_size + 1), state_size],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

        w_h = tf.get_variable(name="w_h", shape=[state_size],
                              initializer=tf.random_uniform_initializer(dtype=tf.float32))

    def conv_r(padded_matrix, r):
        """
        Extract r context columns around each column and concatenate
        :param padded_matrix: batch_size x L+(2*r) x 2*state_size
        :param r: context size
        :return:
        """
        # gather indices of padded
        time_major_matrix = tf.transpose(padded_matrix,
                                         [1, 2, 0])  # time-major  -> L x 2*state_size x batch_size
        contexts = []
        for j in np.arange(r, L + r):
            # extract 2r+1 rows around i for each batch
            context_j = time_major_matrix[j - r:j + r + 1, :, :]  # 2*r+1 x 2*state_size x batch_size
            # concatenate
            context_j = tf.reshape(context_j,
                                   [(2 * r + 1) * 2 * state_size, batch_size])  # (2*r+1)*(state_size) x batch_size
            contexts.append(context_j)
        contexts = tf.stack(contexts)  # L x (2*r+1)* 2*(state_size) x batch_size
        batch_major_contexts = tf.transpose(contexts, [2, 0, 1])
        # switch back: batch_size x L x (2*r+1)*2(state_size) (batch-major)
        return batch_major_contexts

    def prepare_tensor(hidstates, sk, padding_col):
        hs = tf.concat([hidstates, sk], 2)
        # add column on right and left, and add context window
        hs = tf.pad(hs, padding_col, "CONSTANT", name="HS_padded")
        hs = conv_r(hs, window_size)  # [batch_size, L, 2*state*(2*window_size + 1)]
        return hs

    def sketch_step(tensor, cum_attention, active_mask, temper):

        def csoftmax(ten, u, mask):
            """
            Compute the constrained softmax (csoftmax);
            See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
            on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)

            :param ten: input tensor
            :param u: cumulative attention see paper
            :param mask: mask with active elements
            :return: distribution
            """

            shape_t = ten.shape
            shape_u = u.shape
            assert shape_u == shape_t

            # mean
            ten = ten - tf.reduce_mean(ten, axis=1, keep_dims=True)

            neg_mask = tf.ones_like(mask) - mask

            # calculate new distribution with attention on distribution 'b'
            Q = tf.exp(ten)
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

        def attention(t):
            before_att = activation(tf.matmul(t, W_hsz) + w_z)
            att = tf.matmul(before_att, v)  # [batch_size, 1]
            return att

        tensor = tf.transpose(tensor, [1, 0, 2])  # [L; batch_size; 2*state_size*(2*window_size + 1)]

        attentions = tf.map_fn(attention, tensor, dtype=tf.float32)  # [L, batch_size, 1]
        attentions = tf.reshape(attentions, [batch_size, L]) - cum_attention*discount_factor  # [batch_size, L]

        U = tf.ones_like(cum_attention) - cum_attention
        constrained_weights, new_mask = csoftmax(attentions, U, active_mask)  # [batch_size, L]

        tensor = tf.transpose(tensor, [1, 0, 2])  # [batch_size; L; 2*state_size*(2*window_size + 1)]

        if not full_model:
            # TODO: check
            cn = tf.reduce_sum(tensor*tf.expand_dims(constrained_weights, [2]), axis=1)  # [batch_size,
            #  2*state_size*(2*window_size + 1)]
            cn = tf.reshape(cn, [batch_size, 2*state_size*(2*window_size + 1)])  # [batch_size,
            #  2*state_size*(2*window_size + 1)]
            s = activation(tf.matmul(cn, W_hh) + w_h)  # [batch_size, state_size]

            s = tf.matmul(tf.expand_dims(constrained_weights, [2]), tf.expand_dims(s, [1]))  # [batch_size, L,
            #  state_size]
        else:
            def out_layer(slice):
                out = activation(tf.matmul(slice, W_hh) + w_h)
                return out

            tensor = tf.transpose(tensor, [1, 0, 2])  # [L; batch_size; 2*state_size*(2*window_size + 1)]
            s = tf.map_fn(out_layer, tensor, dtype=tf.float32)  # [L; batch_size; state_size]
            s = tf.transpose(s, [1, 0, 2])  # [batch_size; L; state_size]
            s = tf.expand_dims(constrained_weights, [2]) * s

        return s, constrained_weights, new_mask

    sketch = tf.zeros(shape=[batch_size, L, state_size], dtype=tf.float32)  # sketch tenzor
    cum_att = tf.zeros(shape=[batch_size, L])  # cumulative attention

    padding_hs_col = tf.constant([[0, 0], [window_size, window_size], [0, 0]], name="padding_hs_col")
    temperature = tf.constant(temperature, dtype=tf.float32, name='attention_temperature')

    mask = tf.ones([batch_size, L])

    # if track_sketch:
    # sketches = []
    # cum_attentions = []
    # masks = []

    for i in range(sketches_num):

        sketch_, cum_att_, mask_i = sketch_step(prepare_tensor(hidden_states, sketch, padding_hs_col),
                                                cum_att, mask, temperature)

        sketch += sketch_
        cum_att += cum_att_
        mask = mask_i

        # sketches.append(sketch)  # list of tensors with shape [batch_size, L, state_size]
        # cum_attentions.append(cum_att)  # list of tensors with shape [batch_size, L]
        # masks.append(mask)

    return sketch, cum_att, mask
    # return sketches, cum_attentions, masks


# testing
state_size = 40
dim_hlayer = 20
window_size = 2
batch_size = 10
activation = tf.nn.tanh
L = 10
sketches_num = 10
discount_factor = 0.3
temperature = 1

hidden_states = tf.placeholder(dtype=tf.float32, shape=[batch_size, L, state_size])
input_tensor = np.random.randn(batch_size, L, state_size).astype(np.float32)
# cum_att = np.zeros((batch_size, L)).astype(np.float32)
# print(input_tensor)

sketch_list, cum_attention, masks = attention_block(hidden_states, state_size, window_size, dim_hlayer, batch_size,
                                                    activation, L, sketches_num, discount_factor, temperature, True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    sketch, cum_att, masks = sess.run([sketch_list, cum_attention, masks], feed_dict={hidden_states: input_tensor})

    print(cum_att[0])
    print(sketch[0])
    print(masks[0])

    # print(input_tensor)
    #
    # print(cum_att[-1][0])
    # print(sketch[-1][0])
    # print(masks[-1][0])

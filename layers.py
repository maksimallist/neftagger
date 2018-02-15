import tensorflow as tf
import numpy as np


# Input block
def stacked_rnn(input_units,
                n_hidden_list,
                cell_type='gru'):
    units = input_units
    for n, n_h in enumerate(n_hidden_list):
        with tf.variable_scope('RNN_layer_' + str(n)):
            if cell_type == 'gru':
                forward_cell = tf.nn.rnn_cell.GRUCell(n_h)
                backward_cell = tf.nn.rnn_cell.GRUCell(n_h)
            elif cell_type == 'lstm':
                forward_cell = tf.nn.rnn_cell.LSTMCell(n_h)
                backward_cell = tf.nn.rnn_cell.LSTMCell(n_h)
            else:
                raise RuntimeError('cell_type must be either gru or lstm')
            (rnn_output_fw, rnn_output_bw), _ = \
                tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                                backward_cell,
                                                units,
                                                dtype=tf.float32)

            # Dense layer on the top
            units = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)
    return units


# Heritable attention block
def heritable_attention_block(hidden_states, state_size, window_size, sketch_dim, dim_hlayer, batch_size,
                              activation, L, sketches_num, discount_factor, temperature, full_model):

    # attention parameter
    v = tf.get_variable(name="v", shape=[dim_hlayer, 1],
                        initializer=tf.random_uniform_initializer(dtype=tf.float32))

    def conv_r(padded_matrix, r):
        """
        Extract r context columns around each column and concatenate
        :param padded_matrix: batch_size x L+(2*r) x 2*state_size
        :param r: context size
        :return:
        """
        # TODO: make it for different shape of tensors
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

    def sketch_step(tensor, cum_attention, temper):

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

                    p = q_list[1] * (1.0 - mass_) / tf.reduce_sum(q_list[1])
                    p_new = tf.dynamic_stitch(condition_indices, [q_list[0], p])

                    # verification of the condition and modification of masks
                    less_mask = tf.cast(tf.less(u, p_new), tf.int32)  # 0 when u bigger than p, 1 when u less than p
                    condition_indices = tf.dynamic_partition(tf.range(tf.shape(p_new)[0]), less_mask,
                                                             2)  # 0 when u bigger
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

        def attention(t):
            att = tf.matmul(t, v)  # [batch_size, 1]
            return att

        before_att = tf.layers.dense(tensor, dim_hlayer, activation=activation)  # [batch_size; L; dim_hlayer]
        before_att = tf.transpose(before_att, [1, 0, 2])  # [L; batch_size; dim_hlayer]

        attentions = tf.map_fn(attention, before_att, dtype=tf.float32)  # [L, batch_size, 1]
        attentions = tf.reshape(attentions, [batch_size, L]) - cum_attention*discount_factor  # [batch_size, L]

        U = tf.ones_like(cum_attention) - cum_attention
        constrained_weights = csoftmax(attentions, U, temper)  # [batch_size, L]

        if not full_model:
            cn = tf.reduce_sum(tensor*tf.expand_dims(constrained_weights, [2]), axis=1)  # [batch_size,
            #  2*state_size*(2*window_size + 1), 1]
            cn = tf.reshape(cn, [batch_size, 2*state_size*(2*window_size + 1)])  # [batch_size,
            #  2*state_size*(2*window_size + 1)]
            s = tf.layers.dense(cn, sketch_dim, activation=activation)  # [batch_size, sketch_dim]

            s = tf.matmul(tf.expand_dims(constrained_weights, [2]), tf.expand_dims(s, [1]))  # [batch_size, L,
            #  sketch_dim]
        else:
            s = tf.layers.dense(tensor, sketch_dim, activation=activation)  # [batch_size; L; sketch_dim]
            s = tf.expand_dims(constrained_weights, [2]) * s  # [batch_size; L; sketch_dim]

        return s, constrained_weights

    sketch = tf.zeros(shape=[batch_size, L, sketch_dim], dtype=tf.float32)  # sketch tenzor
    cum_att = tf.zeros(shape=[batch_size, L])  # cumulative attention

    padding_hs_col = tf.constant([[0, 0], [window_size, window_size], [0, 0]], name="padding_hs_col")
    temperature = tf.constant(temperature, dtype=tf.float32, name='attention_temperature')

    for i in range(sketches_num):

        sketch_, cum_att_ = sketch_step(prepare_tensor(hidden_states, sketch, padding_hs_col),
                                        cum_att, temperature)

        sketch += sketch_
        cum_att += cum_att_

    return sketch, cum_att

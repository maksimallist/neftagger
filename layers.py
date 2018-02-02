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
                              activation, L, sketches_num, discount_factor, temperature, full_model, drop_):

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

    def sketch_step(tensor, cum_attention, active_mask, temper):

        def csoftmax(ten, u, mask, temp):
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

        def attention(t):
            att = tf.matmul(t, v)  # [batch_size, 1]
            return att

        before_att = tf.layers.dense(tensor, dim_hlayer, activation=activation)  # [batch_size; L; dim_hlayer]

        # dropout for preattention tensor
        before_att = tf.nn.dropout(before_att, drop_)

        before_att = tf.transpose(before_att, [1, 0, 2])  # [L; batch_size; dim_hlayer]

        attentions = tf.map_fn(attention, before_att, dtype=tf.float32)  # [L, batch_size, 1]

        # dropout for preattention tensor
        attentions = tf.nn.dropout(attentions, drop_)

        attentions = tf.reshape(attentions, [batch_size, L]) - cum_attention*discount_factor  # [batch_size, L]

        U = tf.ones_like(cum_attention) - cum_attention
        constrained_weights, new_mask = csoftmax(attentions, U, active_mask, temper)  # [batch_size, L]

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

        return s, constrained_weights, new_mask

    sketch = tf.zeros(shape=[batch_size, L, sketch_dim], dtype=tf.float32)  # sketch tenzor
    cum_att = tf.zeros(shape=[batch_size, L])  # cumulative attention

    padding_hs_col = tf.constant([[0, 0], [window_size, window_size], [0, 0]], name="padding_hs_col")
    temperature = tf.constant(temperature, dtype=tf.float32, name='attention_temperature')

    a_mask = tf.ones([batch_size, L])

    for i in range(sketches_num):

        sketch_, cum_att_, mask_i = sketch_step(prepare_tensor(hidden_states, sketch, padding_hs_col),
                                                cum_att, a_mask, temperature)

        sketch += sketch_
        cum_att += cum_att_
        a_mask = mask_i

    return sketch, cum_att

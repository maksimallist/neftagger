import numpy as np
import tensorflow as tf


# Input block (A_Block)
def input_block(x, seq_lens, drop, lstm_units):

    with tf.name_scope("embedding"):
        # dropout on embeddings
        emb_drop = tf.nn.dropout(x, drop)

    with tf.name_scope("Input"):
        fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, state_is_tuple=True)

        with tf.name_scope("bi-lstm"):
            bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, state_is_tuple=True)

            # dropout on lstm
            # TODO make params, input is already dropped out
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, input_keep_prob=1, output_keep_prob=drop)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, input_keep_prob=1, output_keep_prob=drop)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, emb_drop, sequence_length=seq_lens,
                                                         dtype=tf.float32, time_major=False)
            outputs = tf.concat(outputs, 2)
            state_size = 2 * lstm_units  # concat of fw and bw lstm output

        hidden_states = outputs

    return hidden_states, state_size


# Attention block (B_Block)
def attention_block(hidden_states, state_size, window_size, dim_hlayer, batch_size,
                    activation, L, sketches_num, discount_factor):

    with tf.name_scope("sketching"):

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

        def constrained_softmax(input_tensor, b, temp=1.0):
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
            z = tf.reduce_sum(tf.exp(input_tensor/temp), axis=1, keep_dims=True)
            a = tf.exp(input_tensor/temp) * (b/temp) / z
            # a = tf.exp(input_tensor/temp) * b / z
            u = tf.ones_like(b) - b
            t_mask = tf.to_float(tf.less_equal(a, u))
            f_mask = tf.to_float(tf.less(u, a))
            A = a * t_mask
            U = u * f_mask

            csoftmax = A + U

            return csoftmax

        def sketch_step(tensor, cum_attention, hidden_dim):

            bs_split = tf.split(tensor, L, axis=1)
            attentions = []

            W_hh = tf.get_variable(name="W_hh", shape=[2 * state_size * (2 * window_size + 1), state_size],
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=True,
                                                                                    dtype=tf.float32))

            w_h = tf.get_variable(name="w_z", shape=[state_size],
                                  initializer=tf.random_uniform_initializer(dtype=tf.float32))

            v = tf.get_variable(name="v", shape=[hidden_dim, 1],
                                initializer=tf.random_uniform_initializer(dtype=tf.float32))

            W_hsz = tf.get_variable(name="W_hsz", shape=[2 * state_size * (2 * window_size + 1), hidden_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True,
                                                                                     dtype=tf.float32))

            w_z = tf.get_variable(name="w_z", shape=[hidden_dim],
                                  initializer=tf.random_uniform_initializer(dtype=tf.float32))

            for j in xrange(L):
                tensor = tf.squeeze(bs_split[i])
                preattention = activation(tf.matmul(tensor, W_hsz) + w_z)
                attention = tf.matmul(preattention, v)  # [batch_size, 1]
                attentions.append(attention)

            attentions = tf.stack(attentions, axis=1)  # [batch_size, L]
            attentions = attentions - cum_attention*discount_factor
            constrained_weights = constrained_softmax(attentions, cum_attention)

            cn = tf.reduce_sum(tensor*constrained_weights, axis=1)
            S = activation(tf.matmul(cn, W_hh) + w_h)*constrained_weights

            return S, constrained_weights

        sketch = tf.zeros(shape=[batch_size, L, state_size], dtype=tf.float32)  # sketch tenzor
        cum_att = tf.zeros(shape=[batch_size, L])  # cumulative attention
        padding_hs_col = tf.constant([[0, 0], [window_size, window_size], [0, 0]], name="padding_hs_col")
        sketches = []

        def prepare_tensor(hidstates, sk, padding_col):
            hs = tf.concat(2, [hidstates, sk])
            # add column on right and left, and add context window
            hs = tf.pad(hs, padding_col, "CONSTANT", name="HS_padded")
            hs = conv_r(hs, window_size)  # [batch_size, L, 2*state*(2*window_size + 1)]
            return hs

        for i in xrange(sketches_num):
            sketch_, cum_att_ = sketch_step(prepare_tensor(hidden_states, sketch, padding_hs_col), cum_att, dim_hlayer)
            sketch += sketch_
            cum_att += cum_att_
            sketches.append(sketch_)  # list of tensors with shape [batch_size, L]

    return sketches

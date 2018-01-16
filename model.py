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
# TODO write def() - attention block
def attention_block(self, hidden_states, state_size, track_sketches=False):

    with tf.name_scope("sketching"):
        W_hss = tf.get_variable(name="W_hss", shape=[2 * state_size * (2 * self.window_size + 1), state_size],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
        w_s = tf.get_variable(name="w_s", shape=[state_size],
                              initializer=tf.random_uniform_initializer(dtype=tf.float32))
        w_z = tf.get_variable(name="w_z", shape=[self.dim_hlayer],
                              initializer=tf.random_uniform_initializer(dtype=tf.float32))
        v = tf.get_variable(name="v", shape=[self.dim_hlayer, 1],
                            initializer=tf.random_uniform_initializer(dtype=tf.float32))
        W_hsz = tf.get_variable(name="W_hsz", shape=[2 * state_size * (2 * self.window_size + 1), self.dim_hlayer],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
        # dropout within sketch
        # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py
        # #L1078 (inverted dropout)
        W_hss_mask = tf.to_float(tf.less_equal(tf.random_uniform(tf.shape(W_hss)),
                                               self.drop_sketch)) * tf.inv(self.drop_sketch)

        # def softmax_to_hard(tensor):
        #     max_att = tf.reduce_max(tensor, 1)
        #     a_n = tf.cast(tf.equal(tf.expand_dims(max_att, 1), tensor), tf.float32)
        #     return a_n

        # def normalize(tensor):
        #     """
        #     turn a tensor into a probability distribution
        #     :param tensor: 2D tensor
        #     :return:
        #     """
        #     z = tf.reduce_sum(tensor, 1)
        #     t = tensor / tf.expand_dims(z, 1)
        #     return t

        def softmax_with_mask(tensor, tau=1.0):
            """
            compute the softmax including the mask
            the mask is multiplied with exp(x), before the normalization
            :param tensor: 2D
            :param mask: 2D, same shape as tensor
            :param tau: temperature, the cooler the more spiked is distribution
            :return:
            """
            row_max = tf.expand_dims(tf.reduce_max(tensor, 1), 1)
            t_shifted = tensor - row_max
            nom = tf.exp(t_shifted / tau)
            row_sum = tf.expand_dims(tf.reduce_sum(nom, 1), 1)
            softmax = nom / row_sum
            return softmax

        def constrained_softmax(tensor, b, temp=1.0):
            """
            Compute the constrained softmax (csoftmax);
            See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
            on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)

            :param tensor: input tensor
            :param b: cumulative attention see paper
            :param temp: softmax temperature
            :return: distribution
            """

            row_max = tf.expand_dims(tf.reduce_max(tensor, 1), 1)
            t_shifted = tensor - row_max
            nom = tf.exp(t_shifted / tau)
            row_sum = tf.expand_dims(tf.reduce_sum(nom, 1), 1)
            softmax = nom / row_sum

            return softmax

        def z_j(j, padded_matrix):
            """
            Compute attention weight
            :param j:
            :return:
            """
            matrix_sliced = tf.slice(padded_matrix, [0, j, 0],
                                     [self.batch_size, 2 * self.window_size + 1, 2 * state_size])
            matrix_context = tf.reshape(matrix_sliced,
                                        [self.batch_size, 2 * state_size * (2 * self.window_size + 1)],
                                        name="s_context")  # batch_size x 2*state_size*(2*r+1)
            activ = self.activation(tf.matmul(matrix_context, W_hsz) + w_z)
            z_i = tf.matmul(activ, v)
            return z_i

        # def alpha(sequence_len, padded_matrix, b_i, a_previous, discount_factor=0.0, temperature=1.0):
        def alpha(sequence_len, padded_matrix, discount_factor=0.0, temperature=1.0):
            """
            Compute attention weight for all words in sequence in batch
            :return:
            """
            z = []
            for j in np.arange(sequence_len):
                z.append(z_j(j, padded_matrix))
            z_packed = tf.stack(z)  # seq_len, batch_size, 1
            rz = tf.transpose(z_packed, [1, 0, 2])  # batch-major
            rz = tf.reshape(rz, [self.batch_size, sequence_len])
            # subtract cumulative attention
            # a_n = softmax_with_mask(rz, mask, tau=1.0)  # make sure that no attention is spent on padded areas
            a_i = rz
            # a_i = a_i - discount_factor*b_i
            a_i = softmax_with_mask(a_i, tau=temperature)
            # interpolation gate
            # a_i = softmax_with_mask(rz, mask, tau=1.0)
            # g_n = tf.sigmoid(g)  # range (0,1)
            # a_i = tf.mul(g_n, a_previous) + tf.mul((1-g_n), a_i)
            # normalization
            # a_i = normalize(a_i)
            return a_i  # , rz

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
            for j in np.arange(r, self.L + r):
                # extract 2r+1 rows around i for each batch
                context_j = time_major_matrix[j - r:j + r + 1, :, :]  # 2*r+1 x 2*state_size x batch_size
                # concatenate
                context_j = tf.reshape(context_j,
                                       [(2 * r + 1) * 2 * state_size, self.batch_size])  # (2*r+1)*(state_size) x batch_size
                contexts.append(context_j)
            contexts = tf.stack(contexts)  # L x (2*r+1)* 2*(state_size) x batch_size
            batch_major_contexts = tf.transpose(contexts, [2, 0, 1])  # switch back: batch_size x L x (2*r+1)*2(state_size) (batch-major)
            return batch_major_contexts

        # def sketch_step(n_counter, sketch_embedding_matrix, a, b):
        def sketch_step(n_counter, sketch_embedding_matrix, attention_temperature, attention_discount_factor):
            """
            Compute the sketch vector and update the sketch according to attention over words
            :param n_counter:
            :param sketch_embedding_matrix: updated sketch, batch_size x L x 2*state_size (concatenation of H and S)
            :param a:
            :return:
            """
            sketch_embedding_matrix_padded = tf.pad(sketch_embedding_matrix, padding_hs_col,
                                                    "CONSTANT", name="HS_padded")  # add column on right and left

            # beta function
            a_j = alpha(self.L, sketch_embedding_matrix_padded,  # b, a,  # TODO a_j, _ = alpha(...)
                        discount_factor=attention_discount_factor,
                        temperature=attention_temperature)

            # make "hard"
            # a_j = softmax_to_hard(a_j)

            # cumulative attention scores
            # b_j = (tf.cast(n_counter, tf.float32)-1)*b + a_j #rz
            # b_j /= tf.cast(n_counter, tf.float32)
            # b_j = b + a_j
            # b_j = b

            conv = conv_r(sketch_embedding_matrix_padded, self.window_size)  # batch_size x L x 2*state_size*(2*r+1)
            hs_avg = tf.batch_matmul(tf.expand_dims(a_j, [1]), conv)  # batch_size x 1 x 2*state_size*(2*r+1)
            hs_avg = tf.reshape(hs_avg, [self.batch_size, 2 * state_size * (2 * self.window_size + 1)])

            # same dropout for all steps (http://arxiv.org/pdf/1512.05287v3.pdf), mask is ones if no dropout
            ac = tf.matmul(hs_avg, tf.mul(W_hss, W_hss_mask))
            hs_n = self.activation(ac + w_s)  # batch_size x state_size

            sketch_update = tf.batch_matmul(tf.expand_dims(a_j, [2]),
                                            tf.expand_dims(hs_n, [1]))  # batch_size x L x state_size
            embedding_update = tf.zeros(shape=[self.batch_size, self.L, state_size],
                                        dtype=tf.float32)  # batch_size x L x state_size
            sketch_embedding_matrix += tf.concat(2, [embedding_update, sketch_update])
            return n_counter + 1, sketch_embedding_matrix, a_j  # , b_j

        S = tf.zeros(shape=[self.batch_size, self.L, state_size], dtype=tf.float32)
        a_n = tf.zeros(shape=[self.batch_size, self.L])
        HS = tf.concat(2, [hidden_states, S])
        sketches = []
        # b = tf.ones(shape=[batch_size, L], dtype=tf.float32)/L  # cumulative attention
        # b_n = tf.zeros(shape=[batch_size, L], dtype=tf.float32)  # cumulative attention
        # g = tf.Variable(tf.zeros(shape=[L]))

        padding_hs_col = tf.constant([[0, 0], [self.window_size, self.window_size], [0, 0]], name="padding_hs_col")
        n = tf.constant(1, dtype=tf.int32, name="n")

        if track_sketches:  # use for loop (slower, because more memory)
            if self.sketches_num > 0:
                for i in xrange(self.sketches_num):
                    # n, HS, a_n, b_n = sketch_step(n, HS, a_n, b_n)
                    n, HS, a_n = sketch_step(n, HS, a_n)
                    sketch = tf.split(2, 2, HS)[1]
                    # append attention to sketch
                    sketch_attention = tf.concat(2, [sketch, tf.expand_dims(a_n, 2)])
                    # sketch_attention_cumulative = tf.concat(2, [sketch, tf.expand_dims(a_n, 2),
                    #  tf.expand_dims(b_n, 2)])
                    # sketch_attention_cumulative = tf.concat(2, [tf.expand_dims(a_n, 2), tf.expand_dims(b_n, 2)])
                    # sketches.append(sketch_attention_cumulative)
                    sketches.append(sketch_attention)
        else:  # use while loop
            if self.sketches_num > 0:
                (final_n, final_HS, _) = tf.while_loop(  # TODO add argument again if using b_n
                    cond=lambda n_counter, _1, _2: n_counter <= self.sketches_num,  # add argument
                    body=sketch_step,
                    loop_vars=(n, HS, a_n)  # , b_n)
                )
                HS = final_HS

        sketches_tf = tf.stack(sketches)

    with tf.name_scope("scoring"):

        w_p = tf.get_variable(name="w_p", shape=[self.labels_num],
                              initializer=tf.random_uniform_initializer(dtype=tf.float32))

        wsp_size = 2 * state_size

        W_sp = tf.get_variable(name="W_sp", shape=[wsp_size, self.labels_num],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

        # if class_weights is not None:
        #     class_weights = tf.constant(class_weights, name="class_weights")

        def score(hs_j):
            """
            Score the word at index j, returns state vector for this word (column) across batch
            """
            l = tf.matmul(tf.reshape(hs_j, [self.batch_size, 2 * state_size]), W_sp) + w_p

            return l  # batch_size x K

        def score_predict_loss(score_input):
            """
            Predict a label for an input, compute the loss and return label and loss
            """
            [hs_i, y_words] = score_input
            word_label_score = score(hs_i)
            word_label_probs = tf.nn.softmax(word_label_score)
            word_preds = tf.argmax(word_label_probs, 1)
            y_words_full = tf.one_hot(tf.squeeze(y_words), depth=self.labels_num, on_value=1.0, off_value=0.0)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(word_label_score,
                                                                    y_words_full)
            return [word_preds, cross_entropy]


        S = HS

        scores_pred = tf.map_fn(score_predict_loss,
                                [tf.transpose(S, [1, 0, 2]), tf.transpose(y, [1, 0])],
                                dtype=[tf.int64, tf.float32])  # elems are unpacked along dim 0 -> L
        pred_labels = scores_pred[0]
        losses = scores_pred[1]

        losses = tf.reduce_mean(tf.cast(mask, tf.float32) * tf.transpose(losses, [1, 0]), 1)  # masked, batch_size x 1
        losses_reg = losses
        if l2_scale > 0:
            weights_list = [W_hss, W_sp]  # M_src, M_tgt word embeddings not included
            l2_loss = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(l2_scale), weights_list=weights_list)
            losses_reg += l2_loss
        if l1_scale > 0:
            weights_list = [W_hss, W_sp]
            l1_loss = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l1_regularizer(l1_scale), weights_list=weights_list)
            losses_reg += l1_loss

    return losses, losses_reg, pred_labels, M_src, M_tgt, sketches_tf

    # losses, losses_reg, predictions, M_src, M_tgt, sketches_tf = forward(inputs, labels, masks, seq_lens, class_weights)
    # return losses, losses_reg, predictions, M_src, M_tgt, sketches_tf






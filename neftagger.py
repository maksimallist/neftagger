"""
Tensorflow implementation of the neural easy-first-tagger model
- Full-State Model
"""

import numpy as np
import tensorflow as tf
import utils


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

            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, emb_drop, sequence_length=seq_lens,
                                                                   dtype=tf.float32, time_major=False)
            outputs = tf.concat(outputs, 2)
            state_size = 2 * lstm_units  # concat of fw and bw lstm output

    return outputs, final_state, state_size


# def attention_block(hidden_states, state_size, window_size, dim_hlayer, batch_size,
#                     activation, L, sketches_num, discount_factor):
#
#     with tf.name_scope("sketching"):
#
#         def conv_r(padded_matrix, r):
#             """
#             Extract r context columns around each column and concatenate
#             :param padded_matrix: batch_size x L+(2*r) x 2*state_size
#             :param r: context size
#             :return:
#             """
#             # gather indices of padded
#             time_major_matrix = tf.transpose(padded_matrix,
#                                              [1, 2, 0])  # time-major  -> L x 2*state_size x batch_size
#             contexts = []
#             for j in np.arange(r, L + r):
#                 # extract 2r+1 rows around i for each batch
#                 context_j = time_major_matrix[j - r:j + r + 1, :, :]  # 2*r+1 x 2*state_size x batch_size
#                 # concatenate
#                 context_j = tf.reshape(context_j,
#                                        [(2 * r + 1) * 2 * state_size, batch_size])  # (2*r+1)*(state_size) x batch_size
#                 contexts.append(context_j)
#             contexts = tf.stack(contexts)  # L x (2*r+1)* 2*(state_size) x batch_size
#             batch_major_contexts = tf.transpose(contexts, [2, 0, 1])
#             # switch back: batch_size x L x (2*r+1)*2(state_size) (batch-major)
#             return batch_major_contexts
#
#         def constrained_softmax(input_tensor, b, temp=1.0):
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
#             # input_tensor = tf.reduce_mean(input_tensor)
#             z = tf.reduce_sum(tf.exp(input_tensor/temp), axis=1, keep_dims=True)
#             a = tf.exp(input_tensor/temp) * (b/temp) / z
#             # a = tf.exp(input_tensor/temp) * b / z
#             u = tf.ones_like(b) - b
#             t_mask = tf.to_float(tf.less_equal(a, u))
#             f_mask = tf.to_float(tf.less(u, a))
#             A = a * t_mask
#             U = u * f_mask
#
#             csoftmax = A + U
#
#             return csoftmax
#
#         def sketch_step(tensor, cum_attention, hidden_dim):
#
#             with tf.variable_scope("W_hh_scope", reuse=tf.AUTO_REUSE):
#                 W_hh = tf.get_variable(name="W_hh", shape=[2 * state_size * (2 * window_size + 1), state_size],
#                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True,
#                                                                                         dtype=tf.float32))
#             with tf.variable_scope("w_h_scope", reuse=tf.AUTO_REUSE):
#                 w_h = tf.get_variable(name="w_z", shape=[state_size],
#                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))
#
#             with tf.variable_scope("v_scope", reuse=tf.AUTO_REUSE):
#                 v = tf.get_variable(name="v", shape=[1, 1, hidden_dim],
#                                     initializer=tf.random_uniform_initializer(dtype=tf.float32))
#
#             with tf.variable_scope("W_hsz_scope", reuse=tf.AUTO_REUSE):
#                 W_hsz = tf.get_variable(name="W_hsz", shape=[1, 1, 2 * state_size * (2 * window_size + 1), hidden_dim],
#                                         initializer=tf.contrib.layers.xavier_initializer(uniform=True,
#                                                                                          dtype=tf.float32))
#
#             with tf.variable_scope("w_z_scope", reuse=tf.AUTO_REUSE):
#                 w_z = tf.get_variable(name="w_z", shape=[1, 1, hidden_dim],
#                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))
#
#             preattention = activation(tf.matmul(tensor, W_hsz) + w_z)  # [batch_size, L, hid_dim]
#             attention = tf.matmul(preattention, v)  # [batch_size, L]
#
#             attentions = attention - cum_attention*discount_factor  # [batch_size, L]
#             constrained_weights = constrained_softmax(attentions, cum_attention)  # [batch_size, 1]
#
#             cn = tf.matmul(tf.expand_dims(constrained_weights, [1]), tensor)  # [batch_size, 1,
#             #  2*state_size*(2*window_size + 1)]
#             cn = tf.reshape(cn, [batch_size, 2*state_size*(2*window_size + 1)])  # [batch_size,
#             #  2*state_size*(2*window_size + 1)]
#             S = activation(tf.matmul(cn, W_hh) + w_h)  # [batch_size, state_size]
#
#             S = tf.matmul(tf.expand_dims(constrained_weights, [2]), tf.expand_dims(S, [1]))  # [batch_size, L,
#             #  state_size]
#
#             return S, constrained_weights
#
#         def prepare_tensor(hidstates, sk, padding_col):
#             hs = tf.concat([hidstates, sk], 2)
#             # add column on right and left, and add context window
#             hs = tf.pad(hs, padding_col, "CONSTANT", name="HS_padded")
#             hs = conv_r(hs, window_size)  # [batch_size, L, 2*state*(2*window_size + 1)]
#             return hs
#
#         sketch = tf.zeros(shape=[batch_size, L, state_size], dtype=tf.float32)  # sketch tenzor
#         cum_att = tf.zeros(shape=[batch_size, L])  # cumulative attention
#         padding_hs_col = tf.constant([[0, 0], [window_size, window_size], [0, 0]], name="padding_hs_col")
#         sketches = []
#         cum_attentions = []
#
#         for i in xrange(sketches_num):
#             with tf.variable_scope("sketch_loop", reuse=tf.AUTO_REUSE):
#                 sketch_, cum_att_ = sketch_step(prepare_tensor(hidden_states, sketch, padding_hs_col), cum_att,
#                                                 dim_hlayer)
#             sketch += sketch_
#             cum_att += cum_att_
#             sketches.append(sketch_)  # list of tensors with shape [batch_size, L, state_size]
#             cum_attentions.append(cum_att_)  # list of tensors with shape [batch_size, L]
#
#     return sketches, cum_attentions


# Attention block (B_Block)
def attention_block(hidden_states, state_size, window_size, dim_hlayer, batch_size,
                    activation, L, sketches_num, discount_factor, temperature):

    with tf.name_scope("sketching"):

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

        def imagination(tensor, hidden_dim, n, temper, pad_col):

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
                                           [(2 * r + 1) * 2 * state_size,
                                            batch_size])  # (2*r+1)*(state_size) x batch_size
                    contexts.append(context_j)
                contexts = tf.stack(contexts)  # L x (2*r+1)* 2*(state_size) x batch_size
                batch_major_contexts = tf.transpose(contexts, [2, 0, 1])
                # switch back: batch_size x L x (2*r+1)*2(state_size) (batch-major)
                return batch_major_contexts

            def prepare_tensor(hs, padding_col):
                # add column on right and left, and add context window
                hs = tf.pad(hs, padding_col, "CONSTANT", name="HS_padded")
                hs = conv_r(hs, window_size)  # [batch_size, L, 2*state*(2*window_size + 1)]
                return hs

            def attention(t):
                before_att = activation(tf.matmul(t, W_hsz) + w_z)
                att = tf.matmul(before_att, v)  # [batch_size, 1]
                return att

            # Initialize needed parameters and matrix
            sketch_init = tf.zeros(shape=[batch_size, L, state_size], dtype=tf.float32)  # sketch tenzor
            cum_att_init = tf.zeros(shape=[batch_size, L])  # cumulative attention

            with tf.variable_scope("loop_matrix", reuse=tf.AUTO_REUSE):
                W_hh = tf.get_variable(name="W_hh", shape=[2 * state_size * (2 * window_size + 1), state_size],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

                w_h = tf.get_variable(name="w_h", shape=[state_size],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))

                v = tf.get_variable(name="v", shape=[hidden_dim, 1],
                                    initializer=tf.random_uniform_initializer(dtype=tf.float32))

                W_hsz = tf.get_variable(name="W_hsz", shape=[2 * state_size * (2 * window_size + 1), hidden_dim],
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True,
                                                                                     dtype=tf.float32))

                w_z = tf.get_variable(name="w_z", shape=[hidden_dim],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))

            # create queues
            sketch_q = tf.FIFOQueue(1, dtypes=tf.float32)
            cum_att_q = tf.FIFOQueue(1, dtypes=tf.float32)

            # input initialized tenzors in queues
            sketch_q.enqueue(sketch_init)
            cum_att_q.enqueue(cum_att_init)

            # starting sketching "imagination" cycle
            for i in xrange(n):
                # pull out sketch_i and cumulative_attention_i from queues
                sketch_i = sketch_q.dequeue()
                cum_att_i = cum_att_q.dequeue()

                # concat input tenzor with sketch_i
                z = tf.concat([tensor, sketch_i], 2)
                z = prepare_tensor(z, pad_col)

                att_stacked = tf.map_fn(attention, tf.transpose(z, [1, 0, 2]), dtype=tf.float32)
                # print 'shape of stack', att_stacked.shape
                att_stacked = tf.reshape(att_stacked, [batch_size, L])
                # print 'shape of stack', att_stacked.shape
                att_stacked = att_stacked - cum_att_i * discount_factor  # [batch_size, L]
                # print 'shape of stack', att_stacked.shape
                constrained_weights = constrained_softmax(att_stacked, cum_att_i, temper)  # [batch_size, L]

                #
                cn = tf.reduce_sum(z * tf.expand_dims(constrained_weights, [2]), axis=1)  # [batch_size,
                #  2*state_size*(2*window_size + 1)]
                cn = tf.reshape(cn, [batch_size, 2 * state_size * (2 * window_size + 1)])  # [batch_size,
                #  2*state_size*(2*window_size + 1)]
                s = activation(tf.matmul(cn, W_hh) + w_h)  # [batch_size, state_size]

                s_i = tf.matmul(tf.expand_dims(constrained_weights, [2]), tf.expand_dims(s, [1]))  # [batch_size, L,
                #  state_size]

                #
                sketch_next = sketch_i + s_i
                cum_att_next = cum_att_i + constrained_weights

                #
                sketch_q.enqueue(sketch_next)
                cum_att_q.enqueue(cum_att_next)

            sketch_last = sketch_q.dequeue()
            cum_att_last = cum_att_q.dequeue()

            return sketch_last, cum_att_last

        padding_hs_col = tf.constant([[0, 0], [window_size, window_size], [0, 0]], name="padding_hs_col")
        temperature = tf.constant(temperature, dtype=tf.float32, name='attention_temperature')
        # sketches = []
        # cum_attentions = []

        sketch, cumulative_att = imagination(hidden_states, dim_hlayer, sketches_num, temperature, padding_hs_col)

    return sketch, cumulative_att


class NEF():
    def __init__(self, params, tag_vocab):  # class_weights=None, word_vocab_len

        self.L = params['maximum_L']
        self.labels_num = params['labels_num']
        self.embeddings_dim = params['embeddings_dim']
        self.emb_format = params['emb_format']
        self.tag_emb_dim = params['tag_emb_dim']
        # self.word_vocab_len = word_vocab_len
        self.sketches_num = params['sketches_num']
        self.dim_hlayer = params['dim_hlayer']
        self.lstm_units = params['lstm_units']
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']
        self.window_size = params['window']
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer_ = params['optimizer']
        self.activation_func = params['activation']
        optimizer_map = {"sgd": tf.train.GradientDescentOptimizer,
                         "adam": tf.train.AdamOptimizer,
                         "adagrad": tf.train.AdagradOptimizer,
                         "adadelta": tf.train.AdadeltaOptimizer,
                         "rmsprop": tf.train.RMSPropOptimizer,
                         "momemtum": tf.train.MomentumOptimizer}
        self.optimizer = optimizer_map.get(self.optimizer_, tf.train.GradientDescentOptimizer)(self.learning_rate)
        self.embeddings = params['embeddings']
        # self.l2_scale = params['l2_scale']
        # self.l1_scale = params['l1_scale']
        self.drop = params['drop_prob']
        self.mode = params['mode']

        if self.mode not in ['train', 'inf']:
            raise ValueError('Not implemented mode = {}'.format(self.mode))

        self.drop_sketch = params['drop_prob_sketch']
        # self.max_gradient_norm = max_gradient_norm
        # self.update_emb = update_emb
        self.attention_temperature = params['attention_temperature']
        self.attention_discount_factor = params['attention_discount_factor']
        self.max_gradient_norm = params['max_gradient_norm']
        self.l2_scale = params['l2_scale']
        self.l1_scale = params['l1_scale']
        # self.class_weights = class_weights if class_weights is not None else [1. / self.labels_num] * self.labels_num

        self.path = 'Config:\nTask: NER\nNet configuration:\n\tLSTM: bi-LSTM; LSTM units: {0};\n\t\'' \
                    'Hidden layer dim: {1}; Activation Function: {2}\n' \
                    'Other parameters:\n\t' \
                    'Number of lables: {3};\n\tLanguage: Russian;\n\tEmbeddings dimension: {4};\n\t' \
                    'Number of Sketches: {5};\n\tWindow: {6}\n\tBatch size: {7}\n\tLearning rate: {8}\n\t' \
                    'Optimizer: {9};\n\tDropout probability: {10};\n\tSketch dropout probability {11};\n\t' \
                    'Attention tempreture: {12};\n\t' \
                    'Attention discount factor; {13}\n'.format(self.lstm_units,
                                                               self.dim_hlayer,
                                                               self.activation_func,
                                                               self.labels_num,
                                                               self.embeddings_dim,
                                                               self.sketches_num,
                                                               self.window_size,
                                                               self.batch_size,
                                                               self.learning_rate,
                                                               self.optimizer,
                                                               self.drop,
                                                               self.drop_sketch,
                                                               self.attention_temperature,
                                                               self.attention_discount_factor)

        tags = tag_vocab.w2i.keys()  # return_w2i.keys()
        t_emb = np.zeros((len(tags), len(tags)))
        for i, tag in enumerate(tags):
            t_emb[i][i] = 1.
        self.tag_emb = t_emb

        if self.activation_func == 'tanh':
            self.activation = tf.nn.tanh
        elif self.activation_func == "relu":
            self.activation = tf.nn.relu
        elif self.activation_func == "sigmoid":
            self.activation = tf.nn.sigmoid
        else:
            raise NotImplementedError('Not implemented {} activation function'.format(self.activation_func))

        if self.mode == 'inf':
            self.drop = 1
            self.drop_sketch = 1

        # network graph
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.L, self.embeddings_dim])  # Input Text embeddings.
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.L, self.tag_emb_dim])  # Output Tags embeddings.
        self.lenghs = tf.placeholder(tf.int32, [None])  # Lengths of the sentences.

        lstm_out, final_state, state_size = input_block(self.x, self.lenghs, self.drop, self.lstm_units)
        self.sketche, self.cum_attention = attention_block(lstm_out, state_size, self.window_size, self.dim_hlayer,
                                                           self.batch_size, self.activation, self.L, self.sketches_num,
                                                           self.attention_discount_factor,
                                                           self.attention_temperature)

        # self.sketche = self.sketches[-1]  # last sketch
        hs_final = tf.concat([lstm_out, self.sketche], axis=2)  # [batch_size, L, 2*state_size]

        with tf.name_scope("Out"):
            W_out = tf.get_variable(name="W_out", shape=[2*state_size, self.labels_num],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
            b_out = tf.get_variable(name="w_out", shape=[self.labels_num],
                                    initializer=tf.random_uniform_initializer(dtype=tf.float32))

            def score(hs_j):
                """
                Score the word at index j, returns state vector for this word (column) across batch
                """
                l = tf.matmul(tf.reshape(hs_j, [self.batch_size, 2 * state_size]), W_out) + b_out

                return l  # batch_size x K

            def score_predict_loss(score_input):
                """
                Predict a label for an input, compute the loss and return label and loss
                """
                [hs_i, y_words] = score_input
                word_label_score = score(hs_i)
                word_label_probs = tf.nn.softmax(word_label_score)
                word_preds = tf.argmax(word_label_probs, 1)

                # TODO maybe input only sorted number of tags ?
                # y_words_full = tf.one_hot(tf.squeeze(y_words), depth=self.labels_num, on_value=1.0, off_value=0.0)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=word_label_score, logits=y_words)
                return [word_preds, cross_entropy]

            # calculate prediction scores iteratively on "L" axis
            scores_pred = tf.map_fn(score_predict_loss,
                                    [tf.transpose(hs_final, [1, 0, 2]),
                                     tf.cast(tf.transpose(self.y, [1, 0, 2]), tf.float32)],
                                    dtype=[tf.int64, tf.float32])  # elems are unpacked along dim 0 -> L

            self.pred_labels = scores_pred[0]
            self.losses = scores_pred[1]

            # masked, batch_size x 1 (regularization like dropout but mask)
            # losses = tf.reduce_mean(tf.cast(mask, tf.float32) * tf.transpose(losses, [1, 0]), 1)
            self.losses_reg = tf.reduce_mean(tf.transpose(self.losses, [1, 0]), 1)
            # regularization
            with tf.variable_scope('loop_matrix', reuse=tf.AUTO_REUSE):
                W_hh = tf.get_variable(name='W_hh')

            if self.l2_scale > 0:
                weights_list = [W_hh, W_out]  # word embeddings not included
                l2_loss = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l2_regularizer(self.l2_scale), weights_list=weights_list)
                self.losses_reg += l2_loss
            if self.l1_scale > 0:
                weights_list = [W_hh, W_out]
                l1_loss = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l1_regularizer(self.l1_scale), weights_list=weights_list)
                self.losses_reg += l1_loss

        # gradients and update operation for training the model
        if self.mode == 'train':
            train_params = tf.trainable_variables()
            print train_params
            gradients = tf.gradients(tf.reduce_mean(self.losses_reg, 0), train_params)  # batch normalization
            if self.max_gradient_norm > -1:
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.update = self.optimizer.apply_gradients(zip(clipped_gradients, train_params))

            else:
                self.update = self.optimizer.apply_gradients(zip(gradients, train_params))

        self.saver = tf.train.Saver(tf.all_variables())

    def tensorize_example(self, example, mode='train'):

        if mode not in ['train', 'inf']:
            raise ValueError('Not implemented mode = {}'.format(mode))

        sent_num = len(example)
        assert sent_num <= self.batch_size

        lengs = []
        for s in example:
            lengs.append(len(s))

        x = np.zeros((self.batch_size, self.L, self.embeddings_dim))
        y = np.zeros((self.batch_size, self.L, self.embeddings_dim))
        word_emb = utils.load_embeddings(self.embeddings, self.embeddings_dim, self.emb_format)

        for i, sent in enumerate(example):
            for j, z in enumerate(sent):
                x[i, j] = word_emb[z[0]]  # words
                if mode == 'train':
                    y[i, j] = self.tag_emb[z[1]]  # tags

        return x, y, lengs

    def train_op(self, example, sess):
        x, y, lengs = self.tensorize_example(example)

        pred_labels, losses, _ = sess.run([self.pred_labels, self.losses_reg, self.update],
                                          feed_dict={self.x: x, self.y: y, self.lenghs: lengs})

        return pred_labels, losses

    def inference_op(self, example, sess, sketch_=False, all_=False):
        self.mode = 'inf'
        x, y, lengs = self.tensorize_example(example, 'inf')

        if sketch_:
            if all_:
                pred_labels, sketch = sess.run([self.pred_labels, self.sketche],
                                               feed_dict={self.x: x, self.y: y, self.lenghs: lengs})
            else:
                pred_labels, sketch = sess.run([self.pred_labels, self.sketche],
                                               feed_dict={self.x: x, self.y: y, self.lenghs: lengs})
            return pred_labels, sketch
        else:
            pred_labels = sess.run(self.pred_labels, feed_dict={self.x: x, self.y: y, self.lenghs: lengs})

            return pred_labels

    def load(self, sess, path):

        if tf.gfile.Exists(path):
            print "[ Reading model parameters from {} ]".format(path)
            self.saver.restore(sess, path)
        else:
            raise ValueError('No checkpoint in path {}'.format(path))

    def save(self, sess, path, graph=False):
        self.saver.save(sess, path, write_meta_graph=graph)



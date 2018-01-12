"""
Tensorflow implementation of the neural easy-first model
- Full-State Model
"""

import tensorflow as tf
import numpy as np
import time
import sys
import math
import logging
import datetime


# parameters
parameters = dict()
parameters['learning_rate'] = 0.001  # Learning rate.
parameters['optimizer'] = "adam"  # Optimizer [sgd, adam, adagrad, adadelta, momentum]
parameters['batch_size'] = 200  # Batch size to use during training.
parameters['data_dir'] = ''  # Data directory.
parameters['sketch_dir'] = ''  # Directory where sketch dumps are stored
parameters['model_dir'] = ''  # Model directory
parameters['maximum_L'] = 58  # ??? # maximum length of sequences
parameters['activation'] = 'tanh'  # activation function for dense layers in net
parameters['embeddings'] = ''  # path to source language embeddings.
parameters['embeddings_dim'] = 100  # 300 # dimensionality of embeddings
parameters['labels_num'] = 5  # number of labels
parameters['sketches_num'] = 50  # number of sketches
parameters['dim_hlayer'] = 20  # dimensionality of hidden layer
parameters['window'] = 2  # context size
parameters['train'] = True  # training model
parameters['epochs'] = 50  # training epochs
parameters['checkpoint_freq'] = 100  # save model every x epochs
parameters['lstm_units'] = 20  # number of LSTM-RNN encoder units
parameters['attention_discount_factor'] = 0.0  # Attention discount factor
parameters['attention_temperature'] = 1.0  # Attention temperature
parameters['drop_prob'] = 1  # keep probability for dropout during training (1: no dropout)
parameters['drop_prob_sketch'] = 1  # keep probability for dropout during sketching (1: no dropout)
parameters['restore'] = False  # restoring last session from checkpoint
parameters['interactive'] = False  # interactive mode
parameters['track_sketches'] = False  # keep track of the sketches during learning
parameters['sketch_sentence_id'] = 434  # sentence id of sample (dev) to keep track of during sketching
# tf.app.flags.DEFINE_integer("src_vocab_size", 10000, "Vocabulary size.")
# tf.app.flags.DEFINE_integer("tgt_vocab_size", 10000, "Vocabulary size.")
# tf.app.flags.DEFINE_integer("max_train_data_size", 0,
#                             "Limit on the size of training data (0: no limit).")
# tf.app.flags.DEFINE_float("max_gradient_norm", -1,
#                           "maximum gradient norm for clipping (-1: no clipping)")
# tf.app.flags.DEFINE_integer("buckets", 10, "number of buckets")
# tf.app.flags.DEFINE_boolean("update_emb", False, "update the embeddings")
# tf.app.flags.DEFINE_float("l2_scale", 0, "L2 regularization constant")
# tf.app.flags.DEFINE_float("l1_scale", 0, "L1 regularization constant")
# tf.app.flags.DEFINE_integer("threads", 8, "number of threads")

# Input block (A_Block)
# TODO write def() - input block


def input_block(word_vocab_size, emb_dim, max_sent_len,
                lstm_units, keep_prob,
                embeddings=None, activation=tf.nn.tanh,
                update_emb=True, x, seq_lens):

    batch_size = tf.shape(x)[0]
    with tf.name_scope("embedding"):
        if embeddings.table is None:
            # print "Random src embeddings of dimensionality %d" % D
            emb = tf.get_variable(name="emb", shape=[word_vocab_size, emb_dim],
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
        else:
            emb = tf.get_variable(name="emb",
                                  shape=[embeddings.table.shape[0], embeddings.table.shape[1]],
                                  initializer=tf.constant_initializer(embeddings.table),
                                  trainable=update_emb)
            assert emb_dim == len(embeddings.table[0])
            # print "Loading existing embeddings of dimensionality %d" % len(embeddings.table[0])

        # dropout on embeddings
        emb_drop = tf.nn.dropout(emb, keep_prob)  # TODO make param

        # print "embedding size", emb_size
        # x_tgt, x_src = tf.split(2, 2, x)  # split src and tgt part of input
        # emb_tgt = tf.nn.embedding_lookup(M_tgt, x_tgt, name="emg_tgt")  # batch_size x L x window_size x emb_size
        # emb_src = tf.nn.embedding_lookup(M_src, x_src, name="emb_src")  # batch_size x L x window_size x emb_size
        # emb_comb = tf.concat(2, [emb_src, emb_tgt], name="emb_comb")  # batch_size x L x 2*window_size x emb_size
        # emb = tf.reshape(emb_comb, [batch_size, L, window_size * emb_dim],
        #                  name="emb")  # batch_size x L x window_size*emb_size

    with tf.name_scope("hidden"):
        fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, state_is_tuple=True)

        with tf.name_scope("bi-lstm"):
            bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, state_is_tuple=True)

            # dropout on lstm
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, input_keep_prob=1,
                                                    output_keep_prob=keep_prob)  # TODO make params, input is already dropped out
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, input_keep_prob=1, output_keep_prob=keep_prob)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, emb, sequence_length=seq_lens,
                                                         dtype=tf.float32, time_major=False)
            outputs = tf.concat(2, outputs)
            state_size = 2 * lstm_units  # concat of fw and bw lstm output

        H = outputs

    return H, state_size

# Attention block (B_Block)
# TODO write def() - attention block


def attention_block():
        with tf.name_scope("sketching"):
            W_hss = tf.get_variable(name="W_hss", shape=[2 * state_size * (2 * r + 1), state_size],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
            w_s = tf.get_variable(name="w_s", shape=[state_size],
                                  initializer=tf.random_uniform_initializer(dtype=tf.float32))
            w_z = tf.get_variable(name="w_z", shape=[J],
                                  initializer=tf.random_uniform_initializer(dtype=tf.float32))
            v = tf.get_variable(name="v", shape=[J, 1],
                                initializer=tf.random_uniform_initializer(dtype=tf.float32))
            W_hsz = tf.get_variable(name="W_hsz", shape=[2 * state_size * (2 * r + 1), J],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
            # dropout within sketch
            # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py
            # #L1078 (inverted dropout)
            W_hss_mask = tf.to_float(tf.less_equal(tf.random_uniform(tf.shape(W_hss)),
                                                   keep_prob_sketch)) * tf.inv(keep_prob_sketch)

            def softmax_to_hard(tensor):
                max_att = tf.reduce_max(tensor, 1)
                a_n = tf.cast(tf.equal(tf.expand_dims(max_att, 1), tensor), tf.float32)
                return a_n

            def normalize(tensor):
                """
                turn a tensor into a probability distribution
                :param tensor: 2D tensor
                :return:
                """
                z = tf.reduce_sum(tensor, 1)
                t = tensor / tf.expand_dims(z, 1)
                return t

            def softmax_with_mask(tensor, mask, tau=1.0):
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
                nom = tf.exp(t_shifted / tau) * tf.cast(mask, tf.float32)
                row_sum = tf.expand_dims(tf.reduce_sum(nom, 1), 1)
                softmax = nom / row_sum
                return softmax

            def z_j(j, padded_matrix):
                """
                Compute attention weight
                :param j:
                :return:
                """
                matrix_sliced = tf.slice(padded_matrix, [0, j, 0], [batch_size, 2 * r + 1, 2 * state_size])
                matrix_context = tf.reshape(matrix_sliced, [batch_size, 2 * state_size * (2 * r + 1)],
                                            name="s_context")  # batch_size x 2*state_size*(2*r+1)
                activ = activation(tf.matmul(matrix_context, W_hsz) + w_z)
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
                z_packed = tf.pack(z)  # seq_len, batch_size, 1
                rz = tf.transpose(z_packed, [1, 0, 2])  # batch-major
                rz = tf.reshape(rz, [batch_size, sequence_len])
                # subtract cumulative attention
                # a_n = softmax_with_mask(rz, mask, tau=1.0)  # make sure that no attention is spent on padded areas
                a_i = rz
                # a_i = a_i - discount_factor*b_i
                a_i = softmax_with_mask(a_i, mask, tau=temperature)
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
                for j in np.arange(r, L + r):
                    # extract 2r+1 rows around i for each batch
                    context_j = time_major_matrix[j - r:j + r + 1, :, :]  # 2*r+1 x 2*state_size x batch_size
                    # concatenate
                    context_j = tf.reshape(context_j, [(2 * r + 1) * 2 * state_size,
                                                       batch_size])  # (2*r+1)*(state_size) x batch_size
                    contexts.append(context_j)
                contexts = tf.pack(contexts)  # L x (2*r+1)* 2*(state_size) x batch_size
                batch_major_contexts = tf.transpose(contexts, [2, 0,
                                                               1])  # switch back: batch_size x L x (2*r+1)*2(state_size) (batch-major)
                return batch_major_contexts

            # def sketch_step(n_counter, sketch_embedding_matrix, a, b):
            def sketch_step(n_counter, sketch_embedding_matrix, a):
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
                a_j = alpha(L, sketch_embedding_matrix_padded,  # b, a,  # TODO a_j, _ = alpha(...)
                            discount_factor=attention_discount_factor,
                            temperature=attention_temperature)

                # make "hard"
                # a_j = softmax_to_hard(a_j)

                # cumulative attention scores
                # b_j = (tf.cast(n_counter, tf.float32)-1)*b + a_j #rz
                # b_j /= tf.cast(n_counter, tf.float32)
                # b_j = b + a_j
                # b_j = b

                conv = conv_r(sketch_embedding_matrix_padded, r)  # batch_size x L x 2*state_size*(2*r+1)
                hs_avg = tf.batch_matmul(tf.expand_dims(a_j, [1]), conv)  # batch_size x 1 x 2*state_size*(2*r+1)
                hs_avg = tf.reshape(hs_avg, [batch_size, 2 * state_size * (2 * r + 1)])

                # same dropout for all steps (http://arxiv.org/pdf/1512.05287v3.pdf), mask is ones if no dropout
                ac = tf.matmul(hs_avg, tf.mul(W_hss, W_hss_mask))
                hs_n = activation(ac + w_s)  # batch_size x state_size

                sketch_update = tf.batch_matmul(tf.expand_dims(a_j, [2]),
                                                tf.expand_dims(hs_n, [1]))  # batch_size x L x state_size
                embedding_update = tf.zeros(shape=[batch_size, L, state_size],
                                            dtype=tf.float32)  # batch_size x L x state_size
                sketch_embedding_matrix += tf.concat(2, [embedding_update, sketch_update])
                return n_counter + 1, sketch_embedding_matrix, a_j  # , b_j

            S = tf.zeros(shape=[batch_size, L, state_size], dtype=tf.float32)
            a_n = tf.zeros(shape=[batch_size, L])
            HS = tf.concat(2, [H, S])
            sketches = []
            # b = tf.ones(shape=[batch_size, L], dtype=tf.float32)/L  # cumulative attention
            # b_n = tf.zeros(shape=[batch_size, L], dtype=tf.float32)  # cumulative attention
            # g = tf.Variable(tf.zeros(shape=[L]))

            padding_hs_col = tf.constant([[0, 0], [r, r], [0, 0]], name="padding_hs_col")
            n = tf.constant(1, dtype=tf.int32, name="n")

            if track_sketches:  # use for loop (slower, because more memory)
                if N > 0:
                    for i in xrange(N):
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
                if N > 0:
                    (final_n, final_HS, _) = tf.while_loop(  # TODO add argument again if using b_n
                        cond=lambda n_counter, _1, _2: n_counter <= N,  # add argument
                        body=sketch_step,
                        loop_vars=(n, HS, a_n)  # , b_n)
                    )
                    HS = final_HS

            sketches_tf = tf.pack(sketches)

        with tf.name_scope("scoring"):

            w_p = tf.get_variable(name="w_p", shape=[K],
                                  initializer=tf.random_uniform_initializer(dtype=tf.float32))
            if concat:
                wsp_size = 2 * state_size
            else:
                wsp_size = state_size
            W_sp = tf.get_variable(name="W_sp", shape=[wsp_size, K],
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

            if class_weights is not None:
                class_weights = tf.constant(class_weights, name="class_weights")

            def score(hs_j):
                """
                Score the word at index j, returns state vector for this word (column) across batch
                """
                if concat:
                    l = tf.matmul(tf.reshape(hs_j, [batch_size, 2 * state_size]), W_sp) + w_p
                else:
                    l = tf.matmul(tf.reshape(hs_j, [batch_size, state_size]), W_sp) + w_p
                return l  # batch_size x K

            def score_predict_loss(score_input):
                """
                Predict a label for an input, compute the loss and return label and loss
                """
                [hs_i, y_words] = score_input
                word_label_score = score(hs_i)
                word_label_probs = tf.nn.softmax(word_label_score)
                word_preds = tf.argmax(word_label_probs, 1)
                y_words_full = tf.one_hot(tf.squeeze(y_words), depth=K, on_value=1.0, off_value=0.0)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(word_label_score,
                                                                        y_words_full)
                if class_weights is not None:
                    label_weights = tf.reduce_sum(tf.mul(y_words_full, class_weights), 1)
                    cross_entropy = tf.mul(cross_entropy, label_weights)
                return [word_preds, cross_entropy]

            if concat:
                S = HS
            else:
                S = tf.slice(HS, [0, 0, state_size], [batch_size, L, state_size])

            scores_pred = tf.map_fn(score_predict_loss,
                                    [tf.transpose(S, [1, 0, 2]), tf.transpose(y, [1, 0])],
                                    dtype=[tf.int64, tf.float32])  # elems are unpacked along dim 0 -> L
            pred_labels = scores_pred[0]
            losses = scores_pred[1]

            pred_labels = mask * tf.transpose(pred_labels, [1, 0])  # masked, batch_size x L
            losses = tf.reduce_mean(tf.cast(mask, tf.float32) * tf.transpose(losses, [1, 0]),
                                    1)  # masked, batch_size x 1
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

    losses, losses_reg, predictions, M_src, M_tgt, sketches_tf = forward(inputs, labels, masks, seq_lens, class_weights)
    return losses, losses_reg, predictions, M_src, M_tgt, sketches_tf

class NEF():
    def __init__(self, params, class_weights=None, forward_only=False):
        self.labels_num = params['labels_num']
        self.embeddings_dim = params['embeddings_dim']
        self.sketches_num = params['sketches_num']
        self.dim_hlayer = params['dim_hlayer']
        self.window = params['window']
        self.lstm_units = params['lstm_units']
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']
        self.window_size = 2*self.window + 1
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = params['optimizer']
        self.activation_func = params['activation']
        optimizer_map = {"sgd": tf.train.GradientDescentOptimizer,
                         "adam": tf.train.AdamOptimizer,
                         "adagrad": tf.train.AdagradOptimizer,
                         "adadelta": tf.train.AdadeltaOptimizer,
                         "rmsprop": tf.train.RMSPropOptimizer,
                         "momemtum": tf.train.MomentumOptimizer}
        self.optimizer = optimizer_map.get(self.optimizer, tf.train.GradientDescentOptimizer)(self.learning_rate)
        self.embeddings = params['embeddings']
        # self.l2_scale = params['l2_scale']
        # self.l1_scale = params['l1_scale']
        self.drop = params['drop_prob']
        self.drop_sketch = params['drop_prob_sketch']
        # self.max_gradient_norm = max_gradient_norm
        # self.update_emb = update_emb
        self.attention_temperature = params['attention_temperature']
        self.attention_discount_factor = params['attention_discount_factor']

        self.class_weights = class_weights if class_weights is not None else [1. / self.labels_num] * self.labels_num

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
        if self.activation_func == 'tanh':
            self.activation = tf.nn.tanh
        elif self.activation_func == "relu":
            self.activation = tf.nn.relu
        elif self.activation_func == "sigmoid":
            self.activation = tf.nn.sigmoid
        else:
            raise NotImplementedError('Not implemented {} activation function'.format(self.activation_func))

        if forward_only:
            self.drop = 1
            self.drop_sketch = 1

        # prepare input feeds # TODO normal placeholders
        self.inputs = []
        self.labels = []
        self.masks = []
        self.seq_lens = []
        self.losses = []
        self.losses_reg = []
        self.predictions = []
        self.sketches_tfs = []
        self.keep_probs = []
        self.keep_prob_sketches = []
        self.is_trains = []

        for j, max_len in enumerate(self.buckets):

            self.inputs.append(tf.placeholder(tf.int32, shape=[None, max_len, 2 * self.window_size],
                                              name="inputs{0}".format(j)))
            self.labels.append(tf.placeholder(tf.int32, shape=[None, max_len], name="labels{0}".format(j)))
            self.masks.append(tf.placeholder(tf.int64, shape=[None, max_len], name="masks{0}".format(j)))
            self.seq_lens.append(tf.placeholder(tf.int64, shape=[None], name="seq_lens{0}".format(j)))
            self.keep_prob_sketches.append(tf.placeholder(tf.float32, name="keep_prob_sketch{0}".format(j)))
            self.keep_probs.append(tf.placeholder(tf.float32, name="keep_prob{0}".format(j)))
            self.is_trains.append(tf.placeholder(tf.bool, name="is_train{0}".format(j)))

            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                bucket_losses, bucket_losses_reg, bucket_predictions, src_table, tgt_table, sketches = input_block()

                self.losses_reg.append(bucket_losses_reg)
                self.losses.append(bucket_losses)  # list of tensors, one for each bucket
                self.predictions.append(bucket_predictions)  # list of tensors, one for each bucket
                self.sketches_tfs.append(sketches)

        # gradients and update operation for training the model
        if not forward_only:
            params = tf.trainable_variables()
            self.gradient_norms = []
            self.updates = []
            for j in xrange(len(buckets)):
                gradients = tf.gradients(tf.reduce_mean(self.losses_reg[j], 0), params)  # batch normalization
                if max_gradient_norm > -1:
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    update = self.optimizer.apply_gradients(zip(clipped_gradients, params))
                    self.updates.append(update)

                else:
                    self.gradient_norms.append(tf.global_norm(gradients))
                    update = self.optimizer.apply_gradients(zip(gradients, params))
                    self.updates.append(update)

        self.saver = tf.train.Saver(tf.all_variables())


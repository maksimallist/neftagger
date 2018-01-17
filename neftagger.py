"""
Tensorflow implementation of the neural easy-first model
- Full-State Model
"""

import tensorflow as tf
import numpy as np
import utils


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
parameters['labels_num'] = 7  # number of labels
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


class NEF():
    def __init__(self, params, tag_vocab, word_vocab_len, class_weights=None):

        self.L = params['maximum_L']
        self.labels_num = params['labels_num']
        self.embeddings_dim = params['embeddings_dim']
        self.tag_emb_dim = params['tag_emb_dim']
        self.word_vocab_len = word_vocab_len
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
        self.mode = params['mode']
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


        # prepare input feeds
        placeholders = list()
        placeholders.append((tf.float64, [None, None, self.embeddings_dim]))  # Input Text embeddings.
        placeholders.append((tf.int32, [None, None, self.tag_emb_dim]))  # Output Tags embeddings.
        placeholders.append((tf.int32, [None]))  # Lengths of the sentences.

        self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in placeholders]
        dtypes, shapes = zip(*placeholders)
        queue = tf.PaddingFIFOQueue(capacity=1, dtypes=dtypes, shapes=shapes)
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()

        # self.losses = []
        # self.losses_reg = []
        # self.predictions = []
        # self.sketches_tfs = []
        # self.keep_probs = []
        # self.keep_prob_sketches = []


        # # gradients and update operation for training the model
        # if not self.mode == 'inf':
        #     params = tf.trainable_variables()
        #     self.gradient_norms = []
        #     self.updates = []
        #     for j in xrange(len(buckets)):
        #         gradients = tf.gradients(tf.reduce_mean(self.losses_reg[j], 0), params)  # batch normalization
        #         if max_gradient_norm > -1:
        #             clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        #             self.gradient_norms.append(norm)
        #             update = self.optimizer.apply_gradients(zip(clipped_gradients, params))
        #             self.updates.append(update)
        #
        #         else:
        #             self.gradient_norms.append(tf.global_norm(gradients))
        #             update = self.optimizer.apply_gradients(zip(gradients, params))
        #             self.updates.append(update)

        self.saver = tf.train.Saver(tf.all_variables())

    def tensorize_example(self, example, mode='train'):

        sent_num = len(example)
        assert sent_num <= self.batch_size

        lengs = []
        for s in example:
            lengs.append(len(s))

        x = np.zeros((self.batch_size, np.array(lengs).max(), self.embeddings_dim))
        if mode == 'inf':
            y = np.zeros((self.batch_size, np.array(lengs).max(), self.embeddings_dim))
        word_emb = utils.load_embeddings(self.embeddings)

        for i, sent in enumerate(example):
            for j, z in enumerate(sent):
                x[i, j] = word_emb[z[0]]  # words
                if mode == 'inf':
                    y[i, j] = self.tag_emb[z[1]]  # tags

        if mode == 'inf':
            return x, lengs
        else:
            return x, y, lengs

    def start_enqueue_thread(self, train_example, returning=False):
        """
        Initialize queue of tensors that feed one at the input of the model.
        Args:
            train_example: modified dict from agent
            is_training: training flag
            returning: returning flag
        Returns:
            if returning is True, return list of variables:
                [word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids]
        """
        tensorized_example = self.tensorize_example(train_example)
        feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
        self.sess.run(self.enqueue_op, feed_dict=feed_dict)
        if returning:
            return tensorized_example

    # with tf.name_scope("scoring"):
    #
    #     w_p = tf.get_variable(name="w_p", shape=[labels_num],
    #                           initializer=tf.random_uniform_initializer(dtype=tf.float32))
    #
    #     wsp_size = 2 * state_size
    #
    #     W_sp = tf.get_variable(name="W_sp", shape=[wsp_size, labels_num],
    #                            initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
    #
    #     # if class_weights is not None:
    #     #     class_weights = tf.constant(class_weights, name="class_weights")
    #
    #     def score(hs_j):
    #         """
    #         Score the word at index j, returns state vector for this word (column) across batch
    #         """
    #         l = tf.matmul(tf.reshape(hs_j, [batch_size, 2 * state_size]), W_sp) + w_p
    #
    #         return l  # batch_size x K
    #
    #     def score_predict_loss(score_input):
    #         """
    #         Predict a label for an input, compute the loss and return label and loss
    #         """
    #         [hs_i, y_words] = score_input
    #         word_label_score = score(hs_i)
    #         word_label_probs = tf.nn.softmax(word_label_score)
    #         word_preds = tf.argmax(word_label_probs, 1)
    #         y_words_full = tf.one_hot(tf.squeeze(y_words), depth=labels_num, on_value=1.0, off_value=0.0)
    #         cross_entropy = tf.nn.softmax_cross_entropy_with_logits(word_label_score,
    #                                                                 y_words_full)
    #         return [word_preds, cross_entropy]
    #
    #
    #     S = HS
    #
    #     scores_pred = tf.map_fn(score_predict_loss,
    #                             [tf.transpose(S, [1, 0, 2]), tf.transpose(y, [1, 0])],
    #                             dtype=[tf.int64, tf.float32])  # elems are unpacked along dim 0 -> L
    #     pred_labels = scores_pred[0]
    #     losses = scores_pred[1]
    #
    #     losses = tf.reduce_mean(tf.cast(mask, tf.float32) * tf.transpose(losses, [1, 0]), 1)  # masked, batch_size x 1
    #     losses_reg = losses
    #     if l2_scale > 0:
    #         weights_list = [W_hss, W_sp]  # M_src, M_tgt word embeddings not included
    #         l2_loss = tf.contrib.layers.apply_regularization(
    #             tf.contrib.layers.l2_regularizer(l2_scale), weights_list=weights_list)
    #         losses_reg += l2_loss
    #     if l1_scale > 0:
    #         weights_list = [W_hss, W_sp]
    #         l1_loss = tf.contrib.layers.apply_regularization(
    #             tf.contrib.layers.l1_regularizer(l1_scale), weights_list=weights_list)
    #         losses_reg += l1_loss


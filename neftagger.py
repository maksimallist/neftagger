import layers
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
import utils
from os.path import join


class NEF():
    def __init__(self, params, t2i):  # i2t

        # model structure
        self.full_model = params['full_model']
        self.crf = params['crf']
        self.sketches_num = params['sketches_num']

        # embeddings parameters
        self.embeddings = params['embeddings']
        self.embeddings_dim = params['embeddings_dim']
        self.emb_format = params['emb_format']
        self.tag_emb_dim = params['tag_emb_dim']
        self.tag_num = params['labels_num']

        # network architecture
        self.preatt_hid_dim = params['preattention_layer']
        self.sketch_dim = params['sketch_dim']
        self.unit_type = params['unit_tipe']
        self.units = params['number_of_units']
        self.rnn_layers = params['rnn_layers']
        self.drop = params['drop_prob']
        self.drop_sketch = params['drop_prob_sketch']
        self.attention_temperature = params['attention_temperature']
        self.attention_discount_factor = params['attention_discount_factor']
        self.l2_scale = params['l2_scale']
        self.l1_scale = params['l1_scale']
        self.window_size = params['window']
        self.batch_size = params['batch_size']
        self.max_l = params['maximum_L']
        self.activation_func = params['activation']
        if self.activation_func == 'tanh':
            self.activation = tf.nn.tanh
        elif self.activation_func == "relu":
            self.activation = tf.nn.relu
        elif self.activation_func == "sigmoid":
            self.activation = tf.nn.sigmoid
        else:
            raise NotImplementedError('Not implemented {} activation function'.format(self.activation_func))

        # training params
        self.learning_rate = params['learning_rate']
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer_ = params['optimizer']
        optimizer_map = {"sgd": tf.train.GradientDescentOptimizer,
                         "adam": tf.train.AdamOptimizer,
                         "adagrad": tf.train.AdagradOptimizer,
                         "adadelta": tf.train.AdadeltaOptimizer,
                         "rmsprop": tf.train.RMSPropOptimizer,
                         "momemtum": tf.train.MomentumOptimizer}
        self.optimizer = optimizer_map.get(self.optimizer_, tf.train.GradientDescentOptimizer)(self.learning_rate)
        self.max_gradient_norm = params['max_gradient_norm']
        self.mode = params['mode']

        if self.mode not in ['train', 'inf']:
            raise ValueError('Not implemented mode = {}'.format(self.mode))

        if self.mode == 'inf':
            self.drop = 1
            self.drop_sketch = 1

        # loading embeddings and vocabulary
        self.word_emb = utils.load_embeddings(self.embeddings, self.embeddings_dim, self.emb_format)
        self.tag_emb = t2i

        # configuration
        self.config = 'Config:\nTask: NER\nNet configuration:\n\tRNN: Bidirectional RNN;\n\tType of cell: {0};' \
                      '\n\tNumber layers: {1};\n\tNumber units: {2};\n\t' \
                      'Attention layer dim: {3};\n\tSketch dim: {4};\n\tActivation Function: {5}\n' \
                      'Other parameters:\n\t' \
                      'Number of lables: {6};\n\tLanguage: {7};\n\tEmbeddings dimension: {8};\n\t' \
                      'Number of Sketches: {9};\n\tWindow: {10}\n\tUse CRF: {11};\n\tFull Model: {12};\n\t' \
                      'Batch size: {13}\n\tLearning rate: {14}\n\t' \
                      'Optimizer: {15};\n\tDropout probability: {16};\n\tSketch dropout probability {17};\n\t' \
                      'Attention tempreture: {18};\n\t' \
                      'Attention discount factor; {19}\n'.format(self.unit_type,
                                                                 self.rnn_layers,
                                                                 self.units,
                                                                 self.preatt_hid_dim,
                                                                 self.sketch_dim,
                                                                 self.activation_func,
                                                                 self.tag_num,
                                                                 params['language'],
                                                                 self.embeddings_dim,
                                                                 self.sketches_num,
                                                                 self.window_size,
                                                                 self.crf,
                                                                 self.full_model,
                                                                 self.batch_size,
                                                                 self.learning_rate,
                                                                 self.optimizer_,
                                                                 self.drop,
                                                                 self.drop_sketch,
                                                                 self.attention_temperature,
                                                                 self.attention_discount_factor)

        # network graph
        # TODO: make batch_size and max_l None
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.max_l, self.embeddings_dim])  # Input Text embeddings
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.max_l])  # Output Tags embeddings
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.max_l])

        # Input block (A_Block)
        with tf.name_scope("embedding"):
            # dropout on embeddings
            emb_drop = tf.nn.dropout(self.x, self.drop)

        if self.unit_type is None or self.unit_type not in {'lstm', 'gru'}:
            raise RuntimeError('You must specify the type of the cell! It could be either "lstm" or "gru"')
        rnn_out = layers.stacked_rnn(emb_drop, self.units, cell_type=self.unit_type)
        state_size = 2 * self.units[-1]  # concat of fw and bw lstm output

        # Attention block
        self.sketch, self.cum_att_last = layers.heritable_attention_block(rnn_out,
                                                                          state_size,
                                                                          self.window_size,
                                                                          self.sketch_dim,
                                                                          self.preatt_hid_dim,
                                                                          self.batch_size,
                                                                          self.activation,
                                                                          self.max_l,
                                                                          self.sketches_num,
                                                                          self.attention_discount_factor,
                                                                          self.attention_temperature,
                                                                          self.full_model)

        hs_final = tf.concat([rnn_out, self.sketch], axis=2)  # [batch_size, L, 2*state_size]

        # Classifier
        with tf.variable_scope('Classifier'):
            # TODO: maybe need make relu activation or something else
            # TODO: maybe need to parametrize it
            logits = tf.layers.dense(hs_final, self.tag_num, kernel_initializer=xavier_initializer(),
                                     activation=self.activation)

        if self.crf:
            sequence_lengths = tf.reduce_sum(self.mask, axis=1)
            log_likelihood, trainsition_params = tf.contrib.crf.crf_log_likelihood(logits,
                                                                                   tf.cast(self.y, tf.int32),
                                                                                   sequence_lengths)
            loss_tensor = -log_likelihood
            self.predictions = None
        else:
            ground_truth_labels = tf.one_hot(tf.cast(self.y, tf.int32), self.tag_num)
            loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_labels, logits=logits)
            loss_tensor = loss_tensor * self.mask
            self.predictions = tf.argmax(logits, axis=-1)

        self.loss = tf.reduce_mean(loss_tensor)

        # gradients and update operation for training the model
        if self.mode == 'train':
            train_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, train_params)

            if self.max_gradient_norm > -1:
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.update = self.optimizer.apply_gradients(zip(clipped_gradients, train_params))

            else:
                self.update = self.optimizer.apply_gradients(zip(gradients, train_params))

        self.saver = tf.train.Saver(tf.global_variables())

    def tensorize_example(self, example, mode='train'):

        if mode not in ['train', 'inf']:
            raise ValueError('Not implemented mode = {}'.format(mode))

        # sent_num = len(example)
        # assert sent_num <= self.batch_size
        # batch_size = len(example)
        # L = max([len(sent) for sent in example])

        x = np.zeros((self.batch_size, self.max_l, self.embeddings_dim))
        y = np.zeros((self.batch_size, self.max_l))
        x_mask = np.zeros([self.batch_size, self.max_l], dtype=np.float32)

        for i, sent in enumerate(example):
            x_mask[i, :len(sent)] = 1

        for i, sent in enumerate(example):
            for j, z in enumerate(sent):
                x[i, j] = self.word_emb[z[0]]  # words
                y[i, j] = self.tag_emb[z[1]]  # tags

        return x, y, x_mask

    def train_op(self, example, sess):
        x, y, mask = self.tensorize_example(example)

        if self.crf:
            losses, _, cum_att = sess.run([self.loss, self.update, self.cum_att_last],
                                          feed_dict={self.x: x, self.y: y, self.mask: mask})
            return losses
        else:
            pred_labels, losses, _, cum_att = sess.run([self.predictions, self.loss, self.update, self.cum_att_last],
                                                       feed_dict={self.x: x, self.y: y, self.mask: mask})
            return pred_labels, losses

    def inference_op(self, example, sess, sketch_=False, all_=False):
        self.mode = 'inf'
        x, y, mask = self.tensorize_example(example, 'inf')

        if sketch_:
            if all_:
                pred_labels, sketch = sess.run([self.predictions, self.sketch],
                                               feed_dict={self.x: x, self.y: y, self.mask: mask})
            else:
                pred_labels, sketch = sess.run([self.predictions, self.sketch],
                                               feed_dict={self.x: x, self.y: y, self.mask: mask})
            return pred_labels, sketch
        else:
            pred_labels = sess.run(self.predictions, feed_dict={self.x: x, self.y: y, self.mask: mask})

            return pred_labels

    def load(self, sess, path):

        if tf.gfile.Exists(path):
            print("[ Reading model parameters from {} ]".format(path))
            self.saver.restore(sess, path)
        else:
            raise ValueError('No checkpoint in path {}'.format(path))

    def save(self, sess, path):
        self.saver.save(sess, join(path, 'model.max.ckpt'))

import numpy as np
import os
import tensorflow as tf
import time
from os.path import join
from utils import read_dataset, create_vocabulary, load_embeddings
from utils import accuracy, f1s_binary, precision_recall_f1
import sys

name = sys.argv[1]
language = sys.argv[2]
emb_dim = sys.argv[3]

# net parameters
parameters = dict()

if language == 'english':
    if emb_dim == '300':
        parameters['embeddings'] = './embeddings/{}/glove.840B.300d.txt'.format(language)  # path to
        #  source language embeddings.
        parameters['embeddings_dim'] = 300  # dimensionality of embeddings
        parameters['emb_format'] = 'txt'  # binary model or vec file
    else:
        raise ValueError('There are no embeddings with dimension {0}',
                         ' in the directory: {1}'.format(emb_dim, './embeddings/'+language+'/'))
elif language == 'russian':
    if emb_dim == '300':
        parameters['embeddings'] = './embeddings/{}/ft_0.8.3_nltk_yalen_sg_300.bin'.format(language)  # path to
        #  source language embeddings.
        parameters['embeddings_dim'] = 300  # dimensionality of embeddings
        parameters['emb_format'] = 'bin'  # binary model or vec file
    elif emb_dim == '100':
        parameters['embeddings'] = './embeddings/{}/embeddings_lenta_100.vec'.format(language)  # path to
        #  source language embeddings.
        parameters['embeddings_dim'] = 100  # dimensionality of embeddings
        parameters['emb_format'] = 'vec'
    else:
        raise ValueError('There are no embeddings with dimension {0}'
                         ' in the directory: {1}'.format(emb_dim, './embeddings/' + language + '/'))
else:
    raise ValueError('Sorry, {} language is not implemented yet.'.format(language))


parameters['learning_rate'] = 0.001  # Learning rate.
parameters['optimizer'] = "adam"  # Optimizer [sgd, adam, adagrad, adadelta, momentum]
parameters['batch_size'] = 100  # Batch size to use during training.
parameters['activation'] = 'tanh'  # activation function for dense layers in net
parameters['sketches_num'] = 1  # number of sketches
parameters['lstm_units'] = 20  # number of LSTM-RNN encoder units
parameters['dim_hlayer'] = 20  # dimensionality of hidden layer
parameters['window'] = 2  # context size
parameters['attention_discount_factor'] = 0.0  # Attention discount factor
parameters['attention_temperature'] = 1.0  # Attention temperature
parameters['drop_prob'] = 0.3  # keep probability for dropout during training (1: no dropout)
parameters['drop_prob_sketch'] = 1  # keep probability for dropout during sketching (1: no dropout)
parameters["l2_scale"] = 0  # "L2 regularization constant"
parameters["l1_scale"] = 0  # "L1 regularization constant"
parameters["max_gradient_norm"] = -1  # "maximum gradient norm for clipping (-1: no clipping)"
parameters['track_sketches'] = False  #
parameters['full_model'] = True

# TODO: parametrization
parameters['maximum_L'] = 124  # maximum length of sequences
parameters['mode'] = 'train'

train_flag = dict()
train_flag['data_dir'] = './ner/data/{}/'.format(language)  # Data directory.
train_flag['sketch_dir'] = './ner/sketches/{0}/{1}/'.format(language, name)  # Directory where sketch dumps
#  are stored
train_flag['checkpoint_dir'] = './ner/checkpoints/{0}/{1}/'.format(language, name)  # Model directory
train_flag['epochs'] = 20  # training epochs
train_flag['checkpoint_freq'] = 5  # save model every x epochs
train_flag['restore'] = False  # restoring last session from checkpoint
train_flag['interactive'] = False  # interactive mode
train_flag['train'] = True  # training model
train_flag['prediction_path'] = './ner/predictions/{0}/{1}/'.format(language, name)


def batch_generator(sentences, batch_size, k):
    length = len(sentences)
    for i in range(0, length, batch_size):
        yield sentences[i:i+batch_size], k+1


def refactor_data(example, tags, shape):
    n = np.zeros((shape[0], shape[1]))

    for i, s in enumerate(example):
        for j, z in enumerate(s):
            n[i, j] = tags[z[1]]

    return n


def convert(tensor, tags):
    y = []
    for i in range(np.shape(tensor)[0]):
        for j in range(np.shape(tensor)[1]):
            y.append(tags[tensor[i][j]])
        y.append(tags[0])
    return y


# prepare dataset in format:
# data = [[(word1, tag1), (word2, tag2), ...], [(...),(...)], ...]
# list of sentences; sentence is a list if tuples with word and tag
train_data = read_dataset(join(train_flag['data_dir'], 'train.txt'), parameters['maximum_L'], split=False)
tag_vocabulary, i2t = create_vocabulary(train_data)
parameters['labels_num'] = len(tag_vocabulary.keys())  # number of labels
parameters['tag_emb_dim'] = len(tag_vocabulary.keys())


class bi_LSTM():
    def __init__(self, params, t2i, i2t):  # class_weights=None, word_vocab_len

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        self.full_model = params['full_model']
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

        self.word_emb = load_embeddings(self.embeddings, self.embeddings_dim, self.emb_format)

        self.path = 'Config:\nTask: NER\nNet configuration:\n\tLSTM: bi-LSTM; LSTM units: {0};\n\t\' '\
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

        self.tag_emb = t2i

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
        self.x = tf.placeholder(tf.float32, [None, None, self.embeddings_dim])  # Input Text embeddings.
        self.y = tf.placeholder(tf.int32, [None, None])  # Output Tags embeddings.
        # self.x = tf.placeholder(tf.float32, [self.batch_size, self.L, self.embeddings_dim])  # Input Text embeddings.
        # self.y = tf.placeholder(tf.int32, [self.batch_size, self.L, self.tag_emb_dim])  # Output Tags embeddings.

        # Input block (A_Block)
        with tf.name_scope("embedding"):
            # dropout on embeddings
            emb_drop = tf.nn.dropout(self.x, self.drop)

        with tf.name_scope("Input"):
            fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_units, state_is_tuple=True)

            with tf.name_scope("bi-lstm"):
                bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_units, state_is_tuple=True)

                # dropout on lstm
                # TODO make params, input is already dropped out
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, input_keep_prob=1, output_keep_prob=self.drop)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, input_keep_prob=1, output_keep_prob=self.drop)

                outputs, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, emb_drop, dtype=tf.float32,
                                                                       time_major=False)
                outputs = tf.concat(outputs, 2)  # [batch_size, L, state]
                state_size = 2 * self.lstm_units  # concat of fw and bw lstm output

        with tf.name_scope("Out"):
            W_out = tf.get_variable(name="W_out", shape=[state_size, self.labels_num],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
            b_out = tf.get_variable(name="w_out", shape=[self.labels_num],
                                    initializer=tf.random_uniform_initializer(dtype=tf.float32))

            def score(hs_j):
                """
                Score the word at index j, returns state vector for this word (column) across batch
                """
                # l = tf.matmul(tf.reshape(hs_j, [self.batch_size, state_size]), W_out) + b_out
                l = tf.matmul(hs_j, W_out) + b_out

                return l  # [batch_size; labels_num]

            def score_predict_loss(score_input):
                """
                Predict a label for an input, compute the loss and return label and loss
                """
                [hs_i, y_words] = score_input
                word_label_score = score(hs_i)
                # word_label_score = tf.matmul(hs_i, W_out) + b_out
                word_label_probs = tf.nn.softmax(word_label_score)
                word_preds = tf.argmax(word_label_probs, 1)
                word_preds = tf.cast(word_preds, dtype=tf.float32)

                # TODO maybe input only sorted number of tags ?
                y_words_full = tf.one_hot(tf.squeeze(y_words), depth=self.labels_num, on_value=1.0, off_value=0.0,
                                          dtype=tf.float32)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_words_full, logits=word_label_score)
                return [word_preds, cross_entropy]

            # calculate prediction scores iteratively on "L" axis

            outputs = tf.transpose(outputs, [1, 0, 2])  # [L; batch_size; state]
            y = tf.transpose(self.y, [1, 0])  # [L; batch_size]

            scores_pred = tf.map_fn(score_predict_loss, [outputs, y], dtype=[tf.float32, tf.float32])  # why tf.int64 ?

            self.pred_labels = tf.transpose(scores_pred[0], [1, 0])
            # self.pred_labels = tf.transpose(self.pred_labels, [1, 0])
            self.losses = scores_pred[1]  # [batch_size]

            # masked, batch_size x 1 (regularization like dropout but mask)
            # losses = tf.reduce_mean(tf.cast(mask, tf.float32) * tf.transpose(losses, [1, 0]), 1)
            self.losses_reg = tf.reduce_mean(self.losses) # scalar

            # gradients and update operation for training the model
            # if self.mode == 'train':
            #     train_params = tf.trainable_variables()
            #
            #     # self.losses_reg = tf.reduce_mean(self.losses_reg, 0)  # scalar
            #     gradients = tf.gradients(self.losses_reg, train_params)  # batch normalization
            #     if self.max_gradient_norm > -1:
            #         clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            #         self.update = self.optimizer.apply_gradients(zip(clipped_gradients, train_params))
            #
            #     else:
            #         self.update = self.optimizer.apply_gradients(zip(gradients, train_params))
            self.update = self.optimizer.minimize(self.losses_reg)

            self.saver = tf.train.Saver(tf.global_variables())

    def tensorize_example(self, example, mode='train'):

        if mode not in ['train', 'inf']:
            raise ValueError('Not implemented mode = {}'.format(mode))

        # sent_num = len(example)
        # assert sent_num <= self.batch_size
        batch_size = len(example)
        L = max([len(sent) for sent in example])

        x = np.zeros((self.batch_size, self.L, self.embeddings_dim))
        y = np.zeros((self.batch_size, self.L))

        # x_mask = np.zeros([batch_size, L], dtype=np.float32)
        # for n in range(batch_size):
        #     x_mask[n, :len(example[n])] = 1

        for i, sent in enumerate(example):
            for j, z in enumerate(sent):
                x[i, j] = self.word_emb[z[0]]  # words
                if mode == 'train':
                    y[i, j] = self.tag_emb[z[1]]  # tags

        return x, y

    def train_op(self, example, sess):
        x, y = self.tensorize_example(example)

        pred_labels, losses, _ = sess.run([self.pred_labels, self.losses_reg, self.update],
                                          feed_dict={self.x: x, self.y: y})

        # print(cum_att, '\n')
        # print(pred_labels, '\n')

        return pred_labels, losses

    def inference_op(self, example, sess, sketch_=False, all_=False):
        self.mode = 'inf'
        x, y = self.tensorize_example(example, 'inf')

        if sketch_:
            if all_:
                pred_labels = sess.run(self.pred_labels, feed_dict={self.x: x, self.y: y})
            else:
                pred_labels = sess.run(self.pred_labels, feed_dict={self.x: x, self.y: y})
            return pred_labels
        else:
            pred_labels = sess.run(self.pred_labels, feed_dict={self.x: x, self.y: y})

            return pred_labels

    def load(self, sess, path):

        if tf.gfile.Exists(path):
            print("[ Reading model parameters from {} ]".format(path))
            self.saver.restore(sess, path)
        else:
            raise ValueError('No checkpoint in path {}'.format(path))

    def save(self, sess, path, graph=False):
        self.saver.save(sess, path, write_meta_graph=graph)


def train(generator, param, flags):

    with tf.Session() as sess:
        # create model
        model = bi_LSTM(param, tag_vocabulary, i2t)

        # print config
        print(model.path)
        sess.run(tf.global_variables_initializer())

        # start learning
        start_learning = time.time()
        for e in range(flags['epochs']):
            gen = generator(train_data, param['batch_size'], 0)

            train_predictions = []
            train_true = []
            m_pred = []
            m_true = []
            start = time.time()

            for data, i in gen:
                pred_labels, losses = model.train_op(data, sess)
                train_predictions.extend(pred_labels)
                mpr = convert(pred_labels, i2t)
                m_pred.extend(mpr)

                y = refactor_data(data, tag_vocabulary, [param['batch_size'], param['maximum_L']])
                train_true.extend(y)
                mtr = convert(y, i2t)
                m_true.extend(mtr)
                # TODO fix it
                if i % 20:
                    print('[ Epoch {0}; Loss: {1} ]'.format(e, losses))

            result = dict()
            result['accuracy'] = accuracy(train_true, train_predictions)
            result['f1_nof1'], result['f1_nof2'] = f1s_binary(train_true, train_predictions)
            print('\n{}'.format(result))

            conllf1 = precision_recall_f1(m_true, m_pred)

            if e % flags['checkpoint_freq'] == 0:
                if not os.path.isdir(flags['checkpoint_dir']):
                    os.makedirs(flags['checkpoint_dir'])
                model.save(sess, flags['checkpoint_dir'])

        print('[ End. Global Time: {} ]\n'.format(time.time() - start_learning))

    tf.reset_default_graph()

    return None


def main(_):

    train(batch_generator, parameters, train_flag)
    # test(batch_generator, parameters, train_flag, train_flag['checkpoint_dir'])
    # doc_inference(join(train_flag['data_dir'], 'russian_dev.txt'), batch_generator, parameters,
    #               train_flag, train_flag['checkpoint_dir'])


if __name__ == "__main__":
    tf.app.run()

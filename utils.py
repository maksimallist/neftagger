# from collections import defaultdict
# from itertools import count
from collections import Counter
import fasttext
import numpy as np
import collections
import tensorflow as tf


def read_dataset(fname, maximum_sentence_length=-1, read_ordering=False, split=True):
    sent = []
    sent_tag = []
    ordering = None
    sentences = []
    tags = []
    orderings = []

    for line in open(fname, 'r'):
        if ordering is None and read_ordering:
            line = line.lstrip('#')
            line = line.strip().split(' ')

            ordering = [int(index) for index in line]  # it can use only for specials files with other format

            continue
        else:
            line = line.strip().split()
        if not line:
            if sent and (maximum_sentence_length < 0 or
                         len(sent) < maximum_sentence_length):
                if read_ordering:
                    sentences.append(sent)
                    orderings.append(ordering)
                else:
                    sentences.append(sent)
            sent = []
            ordering = None

            if sent_tag and (maximum_sentence_length < 0 or
                             len(sent_tag) < maximum_sentence_length):
                tags.append(sent_tag)

            sent_tag = []
        else:
            w, t = line[0], line[-1]
            if split:
                sent.append(w)
                sent_tag.append(t)
            else:
                sent.append((w, t))
    if read_ordering and split:
        return sentences, tags, orderings
    elif not read_ordering and split:
        return sentences, tags
    else:
        return sentences


def load_embeddings(embedding_path, embedding_size, embedding_format):
    """
    Load emb dict from file, or load pre trained binary fasttext model.
    Args:
        embedding_path: path to the vec file, or binary model
        embedding_size: int, embedding_size
        embedding_format: 'bin' or 'vec'
    Returns: Embeddings dict, or fasttext pre trained model
    """
    print("Loading word embeddings from {}...".format(embedding_path))

    if embedding_format == 'vec':
        default_embedding = np.zeros(embedding_size)
        embedding_dict = collections.defaultdict(lambda: default_embedding)
        skip_first = embedding_format == "vec"
        with open(embedding_path) as f:
            for i, line in enumerate(f.readlines()):
                if skip_first and i == 0:
                    continue
                splits = line.split()
                assert len(splits) == embedding_size + 1
                word = splits[0]
                embedding = np.array([float(s) for s in splits[1:]])
                embedding_dict[word] = embedding
    elif embedding_format == 'bin':
        embedding_dict = fasttext.load_model(embedding_path)
    else:
        raise ValueError('Not supported embeddings format {}'.format(embedding_format))
    print("Done loading word embeddings.")
    return embedding_dict


# class Vocab:
#     def __init__(self, w2i=None):
#         if w2i is None:
#             w2i = defaultdict()
#         self.w2i = dict(w2i)
#         self.i2w = {i: w for w, i in w2i.iteritems()}
#
#     @classmethod
#     def from_corpus(cls, corpus):
#         w2i = defaultdict()
#         for sent in corpus:
#             [w2i[word] for word in sent]
#         return Vocab(w2i)
#
#     def size(self): return len(self.w2i.keys())
#
#     def return_w2i(self):
#         return self.w2i
#
#     def return_i2w(self):
#         return self.i2w


# def create_vocabularies(corpora, word_cutoff=0, lower_case=False):
#     word_counter = Counter()
#     tag_counter = Counter()
#     word_counter['_UNK_'] = word_cutoff + 1
#
#     for corpus in corpora:
#         for s in corpus:
#             for w, t in s:
#                 if lower_case:
#                     word_counter[w.lower()] += 1
#                 else:
#                     word_counter[w] += 1
#                 tag_counter[t] += 1
#
#     words = [w for w in word_counter if word_counter[w] > word_cutoff]
#     tags = [t for t in tag_counter]
#
#     word_vocabulary = Vocab.from_corpus([words])
#     tag_vocabulary = Vocab.from_corpus([tags])
#
#     print('Words: %d' % word_vocabulary.size())
#     print('Tags: %d' % tag_vocabulary.size())
#
#     return word_vocabulary, tag_vocabulary, word_counter
#
#
def create_vocabulary(corpora, word_cutoff=0, lower_case=False):
    word_counter = Counter()
    tag_counter = Counter()
    word_counter['_UNK_'] = word_cutoff + 1
    tags_ = list()
    words_ = list()

    for corpus in corpora:
        for w, t in corpus:
            if lower_case:
                word_counter[w.lower()] += 1
                if w not in words_:
                    words_.append(w)
            else:
                word_counter[w] += 1
            tag_counter[t] += 1
            if t not in tags_:
                tags_.append(t)

    words = [w for w in word_counter if word_counter[w] > word_cutoff]
    tags = [t for t in tag_counter]

    # word_vocabulary = Vocab.from_corpus([words])
    # tag_vocabulary = Vocab.from_corpus([tags])

    # print('Words: %d' % word_vocabulary.size())
    # print('Tags: %d' % tag_vocabulary.size())
    print('Words: %d' % len(word_counter.keys()))
    print('Tags: %d' % len(tags_))

    return words_, tags_, word_counter


def accuracy(y_i, predictions):
    """
    Accuracy of word predictions
    :param y_i:
    :param predictions:
    :return:
    """
    assert len(y_i) == len(predictions)
    correct_words, all = 0.0, 0.0
    for y, y_pred in zip(y_i, predictions):
        # predictions can be shorter than y, because inputs are cropped to specified maximum length
        for y_w, y_pred_w in zip(y, y_pred):
            all += 1
            if y_pred_w == y_w:
                correct_words += 1
    return correct_words/all


def f1s_binary(y_i, predictions):
    """
    F1 scores of two-class predictions
    :param y_i:
    :param predictions:
    :return: F1_class1, F1_class2
    """
    assert len(y_i) == len(predictions)
    fp_1 = 0.0
    tp_1 = 0.0
    fn_1 = 0.0
    tn_1 = 0.0
    for y, y_pred in zip(y_i, predictions):
        for y_w, y_pred_w in zip(y, y_pred):
            if y_w == 0:  # true class is 0
                if y_pred_w == 0:
                    tp_1 += 1
                else:
                    fn_1 += 1
            else:  # true class is 1
                if y_pred_w == 0:
                    fp_1 += 1
                else:
                    tn_1 += 1
    tn_2 = tp_1
    fp_2 = fn_1
    fn_2 = fp_1
    tp_2 = tn_1
    precision_1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) > 0 else 0
    precision_2 = tp_2 / (tp_2 + fp_2) if (tp_2 + fp_2) > 0 else 0
    recall_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0
    recall_2 = tp_2 / (tp_2 + fn_2) if (tp_2 + fn_2) > 0 else 0
    f1_1 = 2 * (precision_1*recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 \
        else 0
    f1_2 = 2 * (precision_2*recall_2) / (precision_2 + recall_2) if (precision_2 + recall_2) > 0 \
        else 0
    return f1_1, f1_2


def attention_block_with_csoftmax(hidden_states, state_size, window_size, dim_hlayer, batch_size,
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

            for j in range(L):
                tensor = tf.squeeze(bs_split[i])
                preattention = activation(tf.matmul(tensor, W_hsz) + w_z)
                attention = tf.matmul(preattention, v)  # [batch_size, 1]
                attentions.append(attention)

            attentions = tf.stack(attentions, axis=1)  # [batch_size, L]
            attentions = attentions - cum_attention*discount_factor
            constrained_weights = constrained_softmax(attentions, cum_attention)  # [batch_size, 1]

            # TODO: check !!!
            # cn = tf.reduce_sum(tensor*constrained_weights, axis=1)  # [batch_size, 1,
            #  2*state_size*(2*window_size + 1)] ?
            cn = tf.matmul(tf.expand_dims(constrained_weights, [1]), tensor)  # [batch_size, 1,
            #  2*state_size*(2*window_size + 1)]
            cn = tf.reshape(cn, [batch_size, 2*state_size*(2*window_size + 1)])  # [batch_size,
            #  2*state_size*(2*window_size + 1)]
            S = activation(tf.matmul(cn, W_hh) + w_h)  # [batch_size, state_size]

            S = tf.matmul(tf.expand_dims(constrained_weights, [2]), tf.expand_dims(S, [1]))  # [batch_size, L,
            #  state_size]

            return S, constrained_weights

        sketch = tf.zeros(shape=[batch_size, L, state_size], dtype=tf.float32)  # sketch tenzor
        cum_att = tf.zeros(shape=[batch_size, L])  # cumulative attention
        padding_hs_col = tf.constant([[0, 0], [window_size, window_size], [0, 0]], name="padding_hs_col")
        sketches = []
        cum_attentions = []

        def prepare_tensor(hidstates, sk, padding_col):
            hs = tf.concat(2, [hidstates, sk])
            # add column on right and left, and add context window
            hs = tf.pad(hs, padding_col, "CONSTANT", name="HS_padded")
            hs = conv_r(hs, window_size)  # [batch_size, L, 2*state*(2*window_size + 1)]
            return hs

        for i in range(sketches_num):
            sketch_, cum_att_ = sketch_step(prepare_tensor(hidden_states, sketch, padding_hs_col), cum_att, dim_hlayer)
            sketch += sketch_
            cum_att += cum_att_
            sketches.append(sketch_)  # list of tensors with shape [batch_size, L, state_size]
            cum_attentions.append(cum_att_)  # list of tensors with shape [batch_size, L]

    return sketches, cum_attentions

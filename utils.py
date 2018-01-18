from collections import defaultdict
from itertools import count
from collections import Counter
import fasttext
import numpy as np
import collections
import tensorflow as tf


def read_dataset(fname, maximum_sentence_length=-1, read_ordering=False):
    sent = []
    ordering = None
    sentences = []
    orderings = []
    for line in file(fname):
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
        else:
            w, t = line[0], line[-1]
            # w, t = line[-2:]
            sent.append((w, t))
    if read_ordering:
        return sentences, orderings
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


class Vocab:
    def __init__(self, w2i=None):
        if w2i is None:
            w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.iteritems()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

    def return_w2i(self):
        return self.w2i

    def return_i2w(self):
        return self.i2w


def create_vocabularies(corpora, word_cutoff=0, lower_case=False):
    word_counter = Counter()
    tag_counter = Counter()
    word_counter['_UNK_'] = word_cutoff + 1

    for corpus in corpora:
        for s in corpus:
            for w, t in s:
                if lower_case:
                    word_counter[w.lower()] += 1
                else:
                    word_counter[w] += 1
                tag_counter[t] += 1

    words = [w for w in word_counter if word_counter[w] > word_cutoff]
    tags = [t for t in tag_counter]

    word_vocabulary = Vocab.from_corpus([words])
    tag_vocabulary = Vocab.from_corpus([tags])

    print('Words: %d' % word_vocabulary.size())
    print('Tags: %d' % tag_vocabulary.size())

    return word_vocabulary, tag_vocabulary, word_counter


def create_vocabulary(corpora, word_cutoff=0, lower_case=False):
    word_counter = Counter()
    tag_counter = Counter()
    word_counter['_UNK_'] = word_cutoff + 1

    for corpus in corpora:
        for w, t in corpus:
            if lower_case:
                word_counter[w.lower()] += 1
            else:
                word_counter[w] += 1
            tag_counter[t] += 1

    words = [w for w in word_counter if word_counter[w] > word_cutoff]
    tags = [t for t in tag_counter]

    word_vocabulary = Vocab.from_corpus([words])
    tag_vocabulary = Vocab.from_corpus([tags])

    print('Words: %d' % word_vocabulary.size())
    print('Tags: %d' % tag_vocabulary.size())

    return word_vocabulary, tag_vocabulary, word_counter


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

            for j in xrange(L):
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

        for i in xrange(sketches_num):
            sketch_, cum_att_ = sketch_step(prepare_tensor(hidden_states, sketch, padding_hs_col), cum_att, dim_hlayer)
            sketch += sketch_
            cum_att += cum_att_
            sketches.append(sketch_)  # list of tensors with shape [batch_size, L, state_size]
            cum_attentions.append(cum_att_)  # list of tensors with shape [batch_size, L]

    return sketches, cum_attentions

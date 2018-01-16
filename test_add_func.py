from collections import defaultdict
from itertools import count
from collections import Counter

import tensorflow as tf
import numpy as np
import utils


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


def load_embeddings(embeddings_file):
    embeddings = {}
    f = open(embeddings_file)
    for line in f:
        line = line.rstrip('\n').rstrip(' ')
        fields = line.split(' ')
        word = fields[0]
        v = [float(val) for val in fields[1:]]
        embeddings[word] = v
    f.close()
    return embeddings


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


# testing
name = '../neural-easy-first/ner/data/russian/russian_train.txt'
sentences = read_dataset(name)
# print('Not ordering: {}\n'.format(sentences))

# emb = load_embeddings('../neural-easy-first/embeddings/russian/embeddings_lenta_100.vec')
# print(emb[sentences[0][0][0]])


word_voc, tag_voc, word_counter = create_vocabulary(sentences)
# print(word_voc.return_w2i().keys())
print(tag_voc.return_w2i().keys())
# print(word_counter)










def tensorize_example(example, batch_size,
                      embeddings_dim, embeddings, tag_emb, mode='train'):
    sent_num = len(example)
    assert sent_num <= batch_size

    lengs = []
    for s in example:
        lengs.append(len(s))

    x = np.zeros((batch_size, np.array(lengs).max(), embeddings_dim))

    word_emb = utils.load_embeddings(embeddings)

    for i, sent in enumerate(example):
        for j, z in enumerate(sent):
            try:
                x[i, j] = word_emb[z[0]]  # words

            except KeyError:
                pass

    return x, lengs


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
            outputs = tf.concat(2, outputs)
            state_size = 2 * lstm_units  # concat of fw and bw lstm output

        hidden_states = outputs

    return hidden_states, state_size


# prepare input feeds
batch_size = 10
train_example = sentences[:batch_size]
path = '../neural-easy-first/embeddings/russian/embeddings_lenta_100.vec'
tags = tag_voc.w2i.keys()  # return_w2i.keys()
t_emb = np.zeros((len(tags), len(tags)))
for i, tag in enumerate(tags):
    t_emb[i][i] = 1.

placeholders = list()
placeholders.append((tf.float64, [None, None, 100]))  # Input Text embeddings.
# placeholders.append((tf.int32, [None, None, 7]))  # Output Tags embeddings.
placeholders.append((tf.int32, [None]))  # Lengths of the sentences.

queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in placeholders]
dtypes, shapes = zip(*placeholders)
# queue = tf.PaddingFIFOQueue(capacity=1, dtypes=dtypes, shapes=shapes)
# enqueue_op = queue.enqueue(queue_input_tensors)
# input_tensors = queue.dequeue()


tensorized_example = tensorize_example(train_example, batch_size, 100, path, t_emb)
h, state_sizes = input_block(placeholders[0], placeholders[1], 0.5, 25)
feed_dict = dict(zip(queue_input_tensors, tensorized_example))
with tf.Session() as sess:
    sess.run(h, feed_dict=feed_dict)
    print(h)


# def csoftmax(z, b):
#     Z = tf.reduce_sum(tf.exp(z))
#     a = tf.tensordot(tf.exp(z), b, axis=0) / Z
#     u = tf.ones_like(b) - b
#     t_mask = tf.less_equal(a, u)
#     f_mask = tf.less(u, a)
#     A = tf.to_int32(a * t_mask)
#     U = tf.to_int32(u * f_mask)
#
#     csoftmax = A + U
#
#     return csoftmax
#
# import tensorflow as tf
# x = tf.constant([5, 4, 3])
# y = tf.constant([1, 7, 3])
# def f1(): return tf.multiply(x, 17)
# def f2(): return tf.add(y, 23)
#
# t_mask = tf.less_equal(x,y)
# f_mask = tf.less(y,x)
# X = x*tf.to_int32(t_mask)
# Y = y*tf.to_int32(f_mask)
# r = X + Y
#
# with tf.Session() as s:
#     s.run(tf.global_variables_initializer())
#     R = s.run(r)
#
# print R

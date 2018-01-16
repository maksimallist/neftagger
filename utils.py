from collections import defaultdict
from itertools import count
from collections import Counter


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


def tensorize_example(example, batch_size,
                      embeddings_dim, embeddings, tag_emb, mode='train'):
    tag = tag_emb.keys()[0]

    sent_num = len(example)
    assert sent_num <= batch_size

    lengs = []
    for s in example:
        lengs.append(len(s))

    x = np.zeros((batch_size, np.array(lengs).max(), embeddings_dim))
    y = np.zeros((batch_size, np.array(lengs).max(), len(tag_emb[tag])))

    for i, sent in enumerate(example):
        for j, z in enumerate(sent):
            try:
                x[i, j] = embeddings[z[0]]  # words
                y[i, j] = tag_emb[z[1]]
            except KeyError:
                pass

    return x, y, lengs


from collections import Counter
import fasttext
import numpy as np
import collections


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

    if embedding_format in ['vec', 'txt']:
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

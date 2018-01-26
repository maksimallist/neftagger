import collections
import fasttext
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from collections import Counter


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
                splits = line.split(' ')
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
    index = dict()
    tags_ = dict()
    k = 0
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
            if t not in tags_.keys():
                tags_[t] = k
                index[k] = t
                k += 1

    words = [w for w in word_counter if word_counter[w] > word_cutoff]
    tags = [t for t in tag_counter]

    # word_vocabulary = Vocab.from_corpus([words])
    # tag_vocabulary = Vocab.from_corpus([tags])

    # print('Words: %d' % word_vocabulary.size())
    # print('Tags: %d' % tag_vocabulary.size())
    print('Words: %d' % len(word_counter.keys()))
    print('Tags: %d' % len(tags_))

    return tags_, index


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


def chunk_finder(current_token, previous_token, tag):
    current_tag = current_token.split('-', 1)[-1]
    previous_tag = previous_token.split('-', 1)[-1]
    if previous_tag != tag:
        previous_tag = 'O'
    if current_tag != tag:
        current_tag = 'O'
    if (previous_tag == 'O' and current_token == 'B-' + tag) or \
            (previous_token == 'I-' + tag and current_token == 'B-' + tag) or \
            (previous_token == 'B-' + tag and current_token == 'B-' + tag) or \
            (previous_tag == 'O' and current_token == 'I-' + tag):
        create_chunk = True
    else:
        create_chunk = False

    if (previous_token == 'I-' + tag and current_token == 'B-' + tag) or \
            (previous_token == 'B-' + tag and current_token == 'B-' + tag) or \
            (current_tag == 'O' and previous_token == 'I-' + tag) or \
            (current_tag == 'O' and previous_token == 'B-' + tag):
        pop_out = True
    else:
        pop_out = False
    return create_chunk, pop_out


def precision_recall_f1(y_true, y_pred, print_results=True, short_report=False, entity_of_interest=None):
    # Find all tags
    tags = set()
    for tag in y_true + y_pred:
        if tag != 'O':
            current_tag = tag[2:]
            tags.add(current_tag)
    tags = sorted(list(tags))

    results = OrderedDict()
    for tag in tags:
        results[tag] = OrderedDict()
    results['__total__'] = OrderedDict()
    n_tokens = len(y_true)
    total_correct = 0
    # Firstly we find all chunks in the ground truth and prediction
    # For each chunk we write starting and ending indices

    for tag in tags:
        count = 0
        true_chunk = list()
        pred_chunk = list()
        y_true = [str(y) for y in y_true]
        y_pred = [str(y) for y in y_pred]
        prev_tag_true = 'O'
        prev_tag_pred = 'O'
        while count < n_tokens:
            yt = y_true[count]
            yp = y_pred[count]

            create_chunk_true, pop_out_true = chunk_finder(yt, prev_tag_true, tag)
            if pop_out_true:
                true_chunk[-1].append(count - 1)
            if create_chunk_true:
                true_chunk.append([count])

            create_chunk_pred, pop_out_pred = chunk_finder(yp, prev_tag_pred, tag)
            if pop_out_pred:
                pred_chunk[-1].append(count - 1)
            if create_chunk_pred:
                pred_chunk.append([count])
            prev_tag_true = yt
            prev_tag_pred = yp
            count += 1

        if len(true_chunk) > 0 and len(true_chunk[-1]) == 1:
            true_chunk[-1].append(count - 1)
        if len(pred_chunk) > 0 and len(pred_chunk[-1]) == 1:
            pred_chunk[-1].append(count - 1)

        # Then we find all correctly classified intervals
        # True positive results
        tp = 0
        for start, stop in true_chunk:
            for start_p, stop_p in pred_chunk:
                if start == start_p and stop == stop_p:
                    tp += 1
                if start_p > stop:
                    break
        total_correct += tp
        # And then just calculate errors of the first and second kind
        # False negative
        fn = len(true_chunk) - tp
        # False positive
        fp = len(pred_chunk) - tp
        if tp + fp > 0:
            precision = tp / (tp + fp) * 100
        else:
            precision = 0
        if tp + fn > 0:
            recall = tp / (tp + fn) * 100
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        results[tag]['precision'] = precision
        results[tag]['recall'] = recall
        results[tag]['f1'] = f1
        results[tag]['n_predicted_entities'] = len(pred_chunk)
        results[tag]['n_true_entities'] = len(true_chunk)
    total_true_entities = 0
    total_predicted_entities = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for tag in results:
        if tag == '__total__':
            continue
        n_pred = results[tag]['n_predicted_entities']
        n_true = results[tag]['n_true_entities']
        total_true_entities += n_true
        total_predicted_entities += n_pred
        total_precision += results[tag]['precision'] * n_pred
        total_recall += results[tag]['recall'] * n_true
        total_f1 += results[tag]['f1'] * n_true
    if total_true_entities > 0:
        accuracy = total_correct / total_true_entities * 100
        total_recall = total_recall / total_true_entities
    else:
        accuracy = 0
        total_recall = 0
    if total_predicted_entities > 0:
        total_precision = total_precision / total_predicted_entities
    else:
        total_precision = 0

    if total_precision + total_recall > 0:
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    else:
        total_f1 = 0

    results['__total__']['n_predicted_entities'] = total_predicted_entities
    results['__total__']['n_true_entities'] = total_true_entities
    results['__total__']['precision'] = total_precision
    results['__total__']['recall'] = total_recall
    results['__total__']['f1'] = total_f1

    if print_results:
        s = 'processed {len} tokens ' \
            'with {tot_true} phrases; ' \
            'found: {tot_pred} phrases;' \
            ' correct: {tot_cor}.\n\n'.format(len=n_tokens,
                                              tot_true=total_true_entities,
                                              tot_pred=total_predicted_entities,
                                              tot_cor=total_correct)

        s += 'precision:  {tot_prec:.2f}%; ' \
             'recall:  {tot_recall:.2f}%; ' \
             'FB1:  {tot_f1:.2f}\n\n'.format(acc=accuracy,
                                             tot_prec=total_precision,
                                             tot_recall=total_recall,
                                             tot_f1=total_f1)

        if not short_report:
            for tag in tags:
                if entity_of_interest is not None:
                    if entity_of_interest in tag:
                        s += '\t' + tag + ': precision:  {tot_prec:.2f}%; ' \
                                          'recall:  {tot_recall:.2f}%; ' \
                                          'F1:  {tot_f1:.2f} ' \
                                          '{tot_predicted}\n\n'.format(tot_prec=results[tag]['precision'],
                                                                       tot_recall=results[tag]['recall'],
                                                                       tot_f1=results[tag]['f1'],
                                                                       tot_predicted=results[tag]['n_predicted_entities'])
                elif tag != '__total__':
                    s += '\t' + tag + ': precision:  {tot_prec:.2f}%; ' \
                                      'recall:  {tot_recall:.2f}%; ' \
                                      'F1:  {tot_f1:.2f} ' \
                                      '{tot_predicted}\n\n'.format(tot_prec=results[tag]['precision'],
                                                                   tot_recall=results[tag]['recall'],
                                                                   tot_f1=results[tag]['f1'],
                                                                   tot_predicted=results[tag]['n_predicted_entities'])
        elif entity_of_interest is not None:
            s += '\t' + entity_of_interest + ': precision:  {tot_prec:.2f}%; ' \
                              'recall:  {tot_recall:.2f}%; ' \
                              'F1:  {tot_f1:.2f} ' \
                              '{tot_predicted}\n\n'.format(tot_prec=results[entity_of_interest]['precision'],
                                                           tot_recall=results[entity_of_interest]['recall'],
                                                           tot_f1=results[entity_of_interest]['f1'],
                                                           tot_predicted=results[entity_of_interest]['n_predicted_entities'])
        print(s)
    return results


def attention_block(hidden_states, state_size, window_size, dim_hlayer, batch_size,
                    activation, L, sketches_num, discount_factor, temperature):

    with tf.variable_scope('loop_matrices', reuse=tf.AUTO_REUSE):
        W_hsz = tf.get_variable(name="W_hsz", shape=[2 * state_size * (2 * window_size + 1), dim_hlayer],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True,
                                                                                 dtype=tf.float32))

        w_z = tf.get_variable(name="w_z", shape=[dim_hlayer],
                              initializer=tf.random_uniform_initializer(dtype=tf.float32))

        v = tf.get_variable(name="v", shape=[dim_hlayer, 1],
                            initializer=tf.random_uniform_initializer(dtype=tf.float32))

        W_hh = tf.get_variable(name="W_hh", shape=[2 * state_size * (2 * window_size + 1), state_size],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

        w_h = tf.get_variable(name="w_h", shape=[state_size],
                              initializer=tf.random_uniform_initializer(dtype=tf.float32))

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

    def prepare_tensor(hidstates, sk, padding_col):
        hs = tf.concat([hidstates, sk], 2)
        # add column on right and left, and add context window
        hs = tf.pad(hs, padding_col, "CONSTANT", name="HS_padded")
        hs = conv_r(hs, window_size)  # [batch_size, L, 2*state*(2*window_size + 1)]
        return hs

    def sketch_step(tensor, cum_attention, temper):

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
            z = tf.reduce_sum(tf.exp(input_tensor / temp), axis=1, keep_dims=True)
            a = tf.exp(input_tensor / temp) * (b / temp) / z
            # a = tf.exp(input_tensor/temp) * b / z
            u = tf.ones_like(b) - b
            t_mask = tf.to_float(tf.less_equal(a, u))
            f_mask = tf.to_float(tf.less(u, a))
            A = a * t_mask
            U = u * f_mask

            csoftmax = A + U

            return csoftmax

        def attention(t):
            before_att = activation(tf.matmul(t, W_hsz) + w_z)
            att = tf.matmul(before_att, v)  # [batch_size, 1]
            return att

        tensor = tf.transpose(tensor, [1, 0, 2])

        attentions = tf.map_fn(attention, tensor, dtype=tf.float32)  # [batch_size, 1, L]
        attentions = tf.reshape(attentions, [batch_size, L]) - cum_attention*discount_factor  # [batch_size, L]
        constrained_weights = constrained_softmax(attentions, cum_attention, temper)  # [batch_size, L]

        tensor = tf.transpose(tensor, [1, 0, 2])
        cn = tf.reduce_sum(tensor*tf.expand_dims(constrained_weights, [2]), axis=1)  # [batch_size,
        #  2*state_size*(2*window_size + 1)]
        cn = tf.reshape(cn, [batch_size, 2*state_size*(2*window_size + 1)])  # [batch_size,
        #  2*state_size*(2*window_size + 1)]
        s = activation(tf.matmul(cn, W_hh) + w_h)  # [batch_size, state_size]

        s = tf.matmul(tf.expand_dims(constrained_weights, [2]), tf.expand_dims(s, [1]))  # [batch_size, L,
        #  state_size]

        return s, constrained_weights

    sketch = tf.zeros(shape=[batch_size, L, state_size], dtype=tf.float32)  # sketch tenzor
    cum_att = tf.zeros(shape=[batch_size, L])  # cumulative attention
    padding_hs_col = tf.constant([[0, 0], [window_size, window_size], [0, 0]], name="padding_hs_col")
    temperature = tf.constant(temperature, dtype=tf.float32, name='attention_temperature')
    sketches = []
    cum_attentions = []

    for i in range(sketches_num):
        sketch_, cum_att_ = sketch_step(prepare_tensor(hidden_states, sketch, padding_hs_col), cum_att,
                                        temperature)
        sketch += sketch_
        cum_att += cum_att_
        sketches.append(sketch_)  # list of tensors with shape [batch_size, L, state_size]
        cum_attentions.append(cum_att_)  # list of tensors with shape [batch_size, L]

    return sketches, cum_attentions

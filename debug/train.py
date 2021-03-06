import os
import sys
import time
from os.path import join

import numpy as np
import tensorflow as tf

from debug.neftagger import NEF
from utils import accuracy, f1s_binary, precision_recall_f1
from utils import read_dataset, create_vocabulary

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
parameters['sketches_num'] = 5  # number of sketches
parameters['lstm_units'] = 20  # number of LSTM-RNN encoder units
parameters['dim_hlayer'] = 20  # dimensionality of hidden layer
parameters['window'] = 2  # context size
parameters['attention_discount_factor'] = 0.1  # Attention discount factor
parameters['attention_temperature'] = 1.0  # Attention temperature
parameters['drop_prob'] = 0.3  # keep probability for dropout during training (1: no dropout)
parameters['drop_prob_sketch'] = 1  # keep probability for dropout during sketching (1: no dropout)
parameters["l2_scale"] = 0  # "L2 regularization constant"
parameters["l1_scale"] = 0  # "L1 regularization constant"
parameters["max_gradient_norm"] = -1  # "maximum gradient norm for clipping (-1: no clipping)"
parameters['track_sketches'] = False  #
parameters['full_model'] = True

# TODO: parametrization
parameters['maximum_L'] = 83  # maximum length of sequences
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
train_data = read_dataset(join(train_flag['data_dir'], 'train.txt'),
                          parameters['maximum_L'], split=False)

if 'dev.txt' not in os.listdir(train_flag['data_dir']):
    dev_data = read_dataset(join(train_flag['data_dir'], 'valid.txt'),
                            parameters['maximum_L'], split=False)
else:
    dev_data = read_dataset(join(train_flag['data_dir'], 'dev.txt'),
                            parameters['maximum_L'], split=False)

test_data = read_dataset(join(train_flag['data_dir'], 'test.txt'),
                         parameters['maximum_L'], split=False)

tag_vocabulary, i2t = create_vocabulary(train_data)
parameters['labels_num'] = len(tag_vocabulary.keys())  # number of labels
parameters['tag_emb_dim'] = len(tag_vocabulary.keys())


def train(generator, param, flags):

    with tf.Session() as sess:
        # create model
        model = NEF(param, tag_vocabulary, i2t)

        # print config
        print(model.path)
        sess.run(tf.global_variables_initializer())

        # start learning
        start_learning = time.time()
        for e in range(flags['epochs']):
            gen = generator(train_data, param['batch_size'], 0)
            gen_dev = generator(dev_data, param['batch_size'], 0)

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
            # print('[ Non official f1 (1): {} ]\n'.format(f1_nof1))
            # print('[ Non official f1 (2): {} ]\n'.format(f1_nof2))
            # print('[ Epoch {0} end; Time: {1} ]\n'.format(e+1, time.time() - start))
            print('\n{}'.format(result))

            conllf1 = precision_recall_f1(m_true, m_pred)

            print('[ Validation on {}: ... ]'.format(join(flags['data_dir'], 'russian_dev.txt')))
            dev_predictions = []
            dev_true = []
            for data, i in gen_dev:
                pred_labels = model.inference_op(data, sess)
                dev_predictions.extend(pred_labels)
                y_dev = refactor_data(data, tag_vocabulary, [param['batch_size'], param['maximum_L']])
                dev_true.extend(y_dev)

            acc = accuracy(dev_true, dev_predictions)
            print('[accuracy = {}]\n'.format(acc))

            f1_nof1, f1_nof2 = f1s_binary(dev_true, dev_predictions)
            print('[ Non official f1 (1): {} ]\n'.format(f1_nof1))
            print('[ Non official f1 (2): {} ]\n'.format(f1_nof2))

            if e % flags['checkpoint_freq'] == 0:
                if not os.path.isdir(flags['checkpoint_dir']):
                    os.makedirs(flags['checkpoint_dir'])
                model.save(sess, flags['checkpoint_dir'])

        print('[ End. Global Time: {} ]\n'.format(time.time() - start_learning))

    tf.reset_default_graph()

    print('Start testing model from checkpoint: {} '.format(flags['checkpoint_dir']))

    with tf.Session() as sess:
        # create model
        model = NEF(param, tag_vocabulary, i2t)

        # print config
        print(model.path)

        model.load(sess, flags['checkpoint_dir'])
        print('[ model was restored ... ]\n'.format(flags['checkpoint_dir']))
        sess.run(tf.global_variables_initializer())

        # start testing
        gen_test = generator(test_data, param['batch_size'])
        test_predictions = []
        test_true = []
        for data in gen_test:
            pred_labels = model.inference_op(data, sess)
            test_predictions.extend(pred_labels)

            y_test = refactor_data(data, tag_vocabulary, [param['batch_size'], param['maximum_L']])
            test_true.extend(y_test)

        acc = accuracy(test_true, test_predictions)
        print('[accuracy = {}]\n'.format(acc))

        f1_nof1, f1_nof2 = f1s_binary(test_true, test_predictions)
        print('[ Non official f1 (1): {} ]'.format(f1_nof1))
        print('[ Non official f1 (2): {} ]'.format(f1_nof2))

    print('[ End of Testing. ]')
    tf.reset_default_graph()

    return None


# def test(generator, param, checkpoint):
#
#     print('Start testing model from checkpoint: {} '.format(checkpoint))
#
#     with tf.Session() as sess:
#         # create model
#         model = NEF(param, tag_vocabulary)
#
#         # print config
#         print(model.path)
#
#         model.load(sess, checkpoint)
#         print('[ model was restored ... ]\n'.format(checkpoint))
#         sess.run(tf.global_variables_initializer())
#
#         # start testing
#         gen = generator(test_data, param['batch_size'])
#         test_predictions = []
#         test_true = []
#         for data in gen:
#             pred_labels = model.inference_op(data, sess)
#             test_predictions.extend(pred_labels)
#
#             y = refactor_data(data, tag_vocabulary, [param['batch_size'], param['maximum_L']])
#             test_true.extend(y)
#
#         acc = accuracy(test_true, test_predictions)
#         print('[accuracy = {}]\n'.format(acc))
#
#         f1_nof1, f1_nof2 = f1s_binary(test_true, test_predictions)
#         print('[ Non official f1 (1): {} ]'.format(f1_nof1))
#         print('[ Non official f1 (2): {} ]'.format(f1_nof2))
#
#     print('[ End of Testing. ]')
#     tf.reset_default_graph()
#
#     return None
#
#
# def doc_inference(adres, generator, param, flags, checkpoint):
#     # TODO: move from there
#     scorer = 'conlleval.txt'
#
#     data = read_dataset(adres, param['maximum_L'], split=False)
#
#     words, tag_vocabulary, word_counter = create_vocabulary(data)
#
#     with tf.Session() as sess:
#         # create model
#         model = NEF(param, tag_vocabulary)
#
#         # print config
#         print(model.path)
#
#         model.load(sess, checkpoint)
#         print('[ model was restored ... ]\n'.format(checkpoint))
#         sess.run(tf.global_variables_initializer())
#
#         # start testing
#         gen = generator(data, param['batch_size'])
#         test_predictions = []
#         for data_ in gen:
#             pred_labels = model.inference_op(data_, sess)
#             test_predictions.extend(pred_labels)
#
#     tf.reset_default_graph()
#
#     print(len(test_predictions[0]))
#
#     # write in file
#     lines = list()
#     sent = list()
#     k_max = 0
#     k = 0
#     with open(adres, 'r') as in_f:
#         for line in in_f:
#             if line != '\n':
#                 sent.append(line[:-1])
#                 k += 1
#             else:
#                 lines.append(sent)
#                 sent = []
#                 if k > k_max:
#                     k_max = k
#                 k = 0
#         in_f.close()
#
#     print(k_max)
#
#     name = adres.split('/')[-1]
#     names = name.split('.')
#     name = names[0] + '_pred.' + names[-1]
#
#     with open(join(flags['prediction_path'], name), 'w') as f:
#         for i, sent in enumerate(lines):
#             for j, t in enumerate(sent):
#                 t = t + ' ' + tag_vocabulary[test_predictions[i][j]] + '\n'
#                 f.write(t)
#             f.write('\n')
#         f.close()
#
#     cmd = 'perl {0} < {1}'.format(scorer, join(flags['prediction_path'], name))
#     os.system(cmd)
#
#     print('END of converting.')
#
#     return None


# def inference(adres, param, flags, checkpoint):
#     data = read_dataset(adres, param['maximum_L'], split=False)
#
#     words, tag_vocabulary, word_counter = create_vocabulary(data)


def main(_):

    train(batch_generator, parameters, train_flag)
    # test(batch_generator, parameters, train_flag, train_flag['checkpoint_dir'])
    # doc_inference(join(train_flag['data_dir'], 'russian_dev.txt'), batch_generator, parameters,
    #               train_flag, train_flag['checkpoint_dir'])


if __name__ == "__main__":
    tf.app.run()

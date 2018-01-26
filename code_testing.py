import numpy as np
import os
import tensorflow as tf
from neftagger import NEF
from os.path import join
from utils import read_dataset, create_vocabulary
from utils import accuracy, f1s_binary, precision_recall_f1


# net parameters
parameters = dict()
parameters['embeddings'] = './embeddings/english/glove.840B.300d.txt'  # path to
#  source language embeddings.
parameters['embeddings_dim'] = 300  # dimensionality of embeddings
parameters['emb_format'] = 'txt'  # binary model or vec file

parameters['learning_rate'] = 0.001  # Learning rate.
parameters['optimizer'] = "adam"  # Optimizer [sgd, adam, adagrad, adadelta, momentum]
parameters['batch_size'] = 100  # Batch size to use during training.
parameters['activation'] = 'tanh'  # activation function for dense layers in net
parameters['sketches_num'] = 5  # number of sketches
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

# TODO: parametrization
parameters['maximum_L'] = 124  # maximum length of sequences
parameters['mode'] = 'train'

train_flag = dict()
train_flag['data_dir'] = './ner/data/english/'  # Data directory.
train_flag['sketch_dir'] = './ner/sketches/english/test/'  # Directory where sketch dumps
#  are stored
train_flag['checkpoint_dir'] = './ner/checkpoints/english/test/'  # Model directory
train_flag['epochs'] = 20  # training epochs
train_flag['checkpoint_freq'] = 5  # save model every x epochs
train_flag['restore'] = False  # restoring last session from checkpoint
train_flag['interactive'] = False  # interactive mode
train_flag['track_sketches'] = False  # keep track of the sketches during learning
train_flag['sketch_sentence_id'] = 434  # sentence id of sample (dev) to keep track of during sketching
train_flag['train'] = True  # training model
train_flag['prediction_path'] = './ner/predictions/english/test/'


def refactor_data(example, tags, shape, conll=False):
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


train_data = read_dataset(join(train_flag['data_dir'], 'train.txt'),
                          parameters['maximum_L'], split=False)

tag_vocabulary, i2t = create_vocabulary(train_data)
parameters['labels_num'] = len(tag_vocabulary.keys())  # number of labels
parameters['tag_emb_dim'] = len(tag_vocabulary.keys())

k = 0
data = train_data[k:k + parameters['batch_size']]
print('data before net: ...')
print(type(data))
print(data, '\n')
print('tags dict: {}'.format(tag_vocabulary))
print('i2t vocab: {}'.format(i2t))

# testing
with tf.Session() as sess:
    # create model
    model = NEF(parameters, tag_vocabulary, i2t)

    # print config
    print(model.path)
    sess.run(tf.global_variables_initializer())

    pred_labels, losses = model.train_op(data, sess)
    print('Out of network: \n')
    print(losses)
    print(pred_labels)

    y = refactor_data(data, tag_vocabulary, [parameters['batch_size'], parameters['maximum_L']])
    print('Refactor input data: \n{}'.format(y))

    acc = accuracy(y, pred_labels)
    print('[accuracy = {}]\n'.format(acc))

    f1_nof1, f1_nof2 = f1s_binary(y, pred_labels)
    print('[ Non official f1 (1): {} ]\n'.format(f1_nof1))
    print('[ Non official f1 (2): {} ]\n'.format(f1_nof2))

    ypred = convert(pred_labels, i2t)
    ytrue = convert(y, i2t)
    # print('true list: \n{}'.format(ytrue))
    # print('pred list: \n{}'.format(ypred))

    result = precision_recall_f1(ytrue, ypred)
    print(result)

tf.reset_default_graph()



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


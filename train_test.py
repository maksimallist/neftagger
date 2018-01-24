import numpy as np
import tensorflow as tf
from neftagger import NEF
import time
from os.path import join
from utils import read_dataset, create_vocabulary
from utils import accuracy, f1s_binary


# net parameters
parameters = dict()
parameters['learning_rate'] = 0.001  # Learning rate.
parameters['optimizer'] = "adam"  # Optimizer [sgd, adam, adagrad, adadelta, momentum]
parameters['batch_size'] = 100  # Batch size to use during training.
parameters['maximum_L'] = 58  # ??? # maximum length of sequences
parameters['activation'] = 'tanh'  # activation function for dense layers in net
parameters['embeddings'] = '../neural-easy-first/embeddings/russian/ft_0.8.3_nltk_yalen_sg_300.bin'  # path to
#  source language embeddings.
parameters['embeddings_dim'] = 300  # 100 # dimensionality of embeddings
parameters['emb_format'] = 'bin'  # binary model or vec file
parameters['labels_num'] = 7  # number of labels
parameters['tag_emb_dim'] = 7  # number of labels
parameters['sketches_num'] = 50  # number of sketches
parameters['dim_hlayer'] = 20  # dimensionality of hidden layer
parameters['window'] = 2  # context size
parameters['lstm_units'] = 20  # number of LSTM-RNN encoder units
parameters['attention_discount_factor'] = 0.0  # Attention discount factor
parameters['attention_temperature'] = 1.0  # Attention temperature
parameters['drop_prob'] = 1  # keep probability for dropout during training (1: no dropout)
parameters['drop_prob_sketch'] = 1  # keep probability for dropout during sketching (1: no dropout)
parameters["l2_scale"] = 0  # "L2 regularization constant"
parameters["l1_scale"] = 0  # "L1 regularization constant"
parameters["max_gradient_norm"] = -1  # "maximum gradient norm for clipping (-1: no clipping)"
parameters['mode'] = 'train'

train_flag = dict()
train_flag['data_dir'] = '../neural-easy-first/ner/data/russian'  # Data directory.
train_flag['sketch_dir'] = '../neural-easy-first/ner/sketches/russian'  # Directory where sketch dumps are stored
train_flag['checkpoint_dir'] = './checkpoints/'  # Model directory
train_flag['epochs'] = 2  # training epochs
train_flag['checkpoint_freq'] = 2  # save model every x epochs
train_flag['restore'] = False  # restoring last session from checkpoint
train_flag['interactive'] = False  # interactive mode
train_flag['track_sketches'] = False  # keep track of the sketches during learning
train_flag['sketch_sentence_id'] = 434  # sentence id of sample (dev) to keep track of during sketching
train_flag['train'] = True  # training model


def batch_generator(sentences, batch_size):
    length = len(sentences)
    for i in range(0, length, batch_size):
        yield sentences[i:i+batch_size]


def refactor_data(example, tags, shape):
    n = np.zeros((shape[0], shape[1]))

    for i, s in enumerate(example):
        for j, z in enumerate(s):
            n[i, j] = int(tags.index(z[1]))

    return n


def train(generator, param, flags):

    # prepare dataset in format:
    # data = [[(word1, tag1), (word2, tag2), ...], [(...),(...)], ...]
    # list of sentences; sentence is a list if tuples with word and tag
    train_data = read_dataset(join(flags['data_dir'], 'russian_train.txt'),
                              param['maximum_L'], split=False)

    # dev_data = read_dataset(join(flags['data_dir'], 'russian_dev.txt'),
    #                         param['maximum_L'], split=False)

    _, tag_vocabulary, _ = create_vocabulary(train_data)

    with tf.Session() as sess:
        # create model
        model = NEF(param, tag_vocabulary)

        # print config
        print(model.path)
        sess.run(tf.global_variables_initializer())

        # start learning
        start_learning = time.time()
        for e in range(flags['epochs']):
            gen = generator(train_data, param['batch_size'])
            train_predictions = []
            train_true = []
            start = time.time()
            for data in gen:
                pred_labels, losses = model.train_op(data, sess)
                train_predictions.extend(pred_labels)

                y = refactor_data(data, tag_vocabulary, [param['batch_size'], param['maximum_L']])
                train_true.extend(y)
                print('[ Epoch {0}; Loss: {1} ]'.format(e, losses))
                # print(pred_labels.shape)
                # print(y.shape)
                # print(pred_labels, '\n')
                # print(y)

            acc = accuracy(train_true, train_predictions)
            print('[accuracy = {}]\n'.format(acc))

            f1_nof1, f1_nof2 = f1s_binary(train_true, train_predictions)
            print('[ Non official f1 (1): {} ]\n'.format(f1_nof1))
            print('[ Non official f1 (2): {} ]'.format(f1_nof2))

            print('[ Epoch {0} end; Time: {1} ]'.format(e+1, time.time() - start))
            if e % flags['checkpoint_freq'] == 0:
                model.save(sess, flags['checkpoint_dir'])

        print('[ End. Global Time: {} ]\n'.format(time.time() - start_learning))

    tf.reset_default_graph()

    return None


def test(generator, param, flags, checkpoint):

    print('Start testing model from checkpoint: {} '.format(checkpoint))

    test_data = read_dataset(join(flags['data_dir'], 'russian_test.txt'),
                             param['maximum_L'], split=False)

    _, tag_vocabulary, _ = create_vocabulary(test_data)

    with tf.Session() as sess:
        # create model
        model = NEF(param, tag_vocabulary)

        # print config
        print(model.path)

        model.load(sess, checkpoint)
        print('[ model was restored ... ]\n'.format(checkpoint))
        sess.run(tf.global_variables_initializer())

        # start testing
        gen = generator(test_data, param['batch_size'])
        test_predictions = []
        test_true = []
        for data in gen:
            pred_labels = model.inference_op(data, sess)
            test_predictions.extend(pred_labels)

            y = refactor_data(data, tag_vocabulary, [param['batch_size'], param['maximum_L']])
            test_true.extend(y)

        acc = accuracy(test_true, test_predictions)
        print('[accuracy = {}]\n'.format(acc))

        f1_nof1, f1_nof2 = f1s_binary(test_true, test_predictions)
        print('[ Non official f1 (1): {} ]'.format(f1_nof1))
        print('[ Non official f1 (2): {} ]'.format(f1_nof2))

    print('[ End of Testing. ]')

    return None


def main(_):
    train(batch_generator, parameters, train_flag)
    test(batch_generator, parameters, train_flag, train_flag['checkpoint_dir'])


if __name__ == "__main__":
    tf.app.run()

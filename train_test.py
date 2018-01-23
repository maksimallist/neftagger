from neftagger import NEF
import tensorflow as tf
import time
from os.path import join
from utils import read_dataset, create_vocabulary
# from utils import accuracy, f1s_binary


# net parameters
parameters = dict()
parameters['learning_rate'] = 0.001  # Learning rate.
parameters['optimizer'] = "adam"  # Optimizer [sgd, adam, adagrad, adadelta, momentum]
parameters['batch_size'] = 200  # Batch size to use during training.
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
train_flag['checkpoint_dir'] = './checkpoints'  # Model directory
train_flag['epochs'] = 50  # training epochs
train_flag['checkpoint_freq'] = 50  # save model every x epochs
train_flag['restore'] = False  # restoring last session from checkpoint
train_flag['interactive'] = False  # interactive mode
train_flag['track_sketches'] = False  # keep track of the sketches during learning
train_flag['sketch_sentence_id'] = 434  # sentence id of sample (dev) to keep track of during sketching
train_flag['train'] = True  # training model


def batch_generator(sentences, batch_size):
    length = len(sentences)
    for i in xrange(0, length, batch_size):
        yield sentences[i:i+batch_size]


def train(generator, param, flags):
    # prepare dataset in format:
    # data = [[(word1, tag1), (word2, tag2), ...], [(...),(...)], ...]
    # list of sentences; sentence is a list if tuples with word and tag
    train_data = read_dataset(join(flags['data_dir'], 'russian_train.txt'),
                              param['maximum_L'], split=False)
    # test_data = read_dataset(join(flags['data_dir'], 'russian_test.txt'),
    #                          param['maximum_L'], split=False)
    # dev_data = read_dataset(join(flags['data_dir'], 'russian_dev.txt'),
    #                         param['maximum_L'], split=False)

    _, tag_vocabulary, _ = create_vocabulary(train_data)

    with tf.Session() as sess:
        # create model
        model = NEF(param, tag_vocabulary)
        # print config
        print model.path
        sess.run(tf.global_variables_initializer())

        # start learning
        start_learning = time.time()
        for e in xrange(flags['epochs']):
            # train_predictions = []
            # train_true = []
            start = time.time()
            for data in generator(train_data, param['batch_size']):
                pred_labels, losses = model.train_op(data, sess)
                # train_predictions.extend(pred_labels)
                # train_true.extend(data[1])
            # print '[accuracy = {}]\n'.format()
            print '[ Epoch {0} end. Time: {1} ]'.format(e, time.time() - start)
            if e % flags['checkpoint_freq'] == 0:
                model.save(sess, flags['checkpoint_dir'])

        print '[ End. Global Time: {} ]'.format(time.time() - start_learning)

    return None


def main(_):
    train(batch_generator, parameters, train_flag)


if __name__ == "__main__":
    tf.app.run()

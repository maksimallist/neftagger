from os.path import join
from utils import read_dataset, create_vocabulary, load_embeddings
import numpy as np


L = 83
datapath = './ner/data/{}/'.format('russian')  # Data directory.

train_data = read_dataset(join(datapath, 'train.txt'), L, split=False)
words, tags = read_dataset(join(datapath, 'train.txt'), L, split=True)
tag_vocabulary, i2t = create_vocabulary(train_data)
lab_num = len(tag_vocabulary.keys())  # number of labels
lab_dim = len(tag_vocabulary.keys())

print('Train data: \n{}'.format(train_data[0]))
print('Split Train data: \n{0}\n{1}'.format(words[0], tags[0]))
print('tags vocabulary: \n{}'.format(tag_vocabulary))
print('I2t: \n{}'.format(i2t))
print('Dimension of tags embeddings: {}'.format(lab_dim))
print('Amount of tags: {}'.format(lab_num))


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


batch_size = 2
# gen = batch_generator(train_data, batch_size, 0)
# for data, j in batch_generator(train_data, batch_size, 0):
#     print('data {0}: {1};'.format(j, data))
#     if j >= 3:
#         break

ex = train_data[0:2]
print(ex)

ref = refactor_data(ex, tag_vocabulary, (batch_size, L))
print('refactoring: \n{}'.format(ref))

# coding tags
y = np.zeros((batch_size, L, lab_dim))
for i, sent in enumerate(ex):
    for j, z in enumerate(sent):
        k = tag_vocabulary[z[1]]
        y[i, j, k] = 1  # tags

print('Matrix of tags: \n{}'.format(y[0]))



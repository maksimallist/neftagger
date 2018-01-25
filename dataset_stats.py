import numpy as np
import utils
import os
from os.path import join


dataset = './ner/data/english/'
data_names = os.listdir(dataset)
lines = 0
lsent = 0
msent = 0

for name in data_names:
    with open(join(dataset, name), 'r') as f:
        for line in f:
            if line.startswith('-DOCSTART-') or line == '\n':
                if lsent > msent:
                    msent = lsent
                lsent = 0
                continue
            else:
                lines += 1
                lsent += 1
        f.close()

print('maximum length of sentence: {}'.format(msent))
print('Number of sentences: {}'.format(lines))

with open(join(dataset, 'info.txt'), 'w') as info:
    info.write('maximum length of the sentence: {}\n'.format(msent))
    info.write('Number sentences: {}\n'.format(lines))

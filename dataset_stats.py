import os
from os.path import join
# import numpy as np
# import utils


def create_tag_vocabulary(file):
    t2i = dict()
    i2t = dict()
    k = 0
    with open(file, 'r') as q:
        for l in q:
            if l.startswith('-DOCSTART-') or line == '\n':
                continue
            else:
                w = l.strip().split()
                if len(w) != 0:
                    t = w[1]
                    if t not in t2i.keys():
                        i2t[k] = t
                        t2i[t] = k
                        k += 1
        q.close()

    return t2i, i2t, len(t2i.keys())


data = './ner/data/'
languages = os.listdir(data)
print('Found {0} directories for languages: {1}.\n'.format(len(languages), languages))

for lang in languages:
    data_names = os.listdir(join(data, lang))
    if len(data_names) == 0:
        print('There are no files in the directory {}.'.format(join(data, lang)))

    lines = 0
    lsent = 0
    msent = 0

    print('Information from {} datset:'.format(lang))
    with open(join(join(data, lang), 'info.txt'), 'w') as info:
        info.write('Information from {} datset:\n'.format(lang))
        for name in data_names:
            if name == 'info.txt':
                continue

            nlines = 0
            nlsent = 0
            nmsent = 0
            with open(join(data, lang, name), 'r') as f:
                for line in f:
                    if line.startswith('-DOCSTART-') or line == '\n':
                        if lsent > msent:
                            msent = lsent
                        lsent = 0

                        if nlsent > nmsent:
                            nmsent = nlsent
                        nlsent = 0

                        continue
                    else:
                        lines += 1
                        lsent += 1
                        nlines += 1
                        nlsent += 1
                f.close()

            t2i, i2t, ntags = create_tag_vocabulary(join(data, lang, name))

            print('File: {}'.format(name))
            print('\t\tmaximum length of sentence: {}'.format(nmsent))
            print('\t\tNumber of sentences: {}'.format(nlines))
            info.write('\tFile: {}\n'.format(name))
            info.write('\t\tmaximum length of sentence: {}\n'.format(nmsent))
            info.write('\t\tNumber of sentences: {}\n'.format(nlines))
            info.write('\t\tNumber of tags: {}'.format(ntags))
            info.write('\t\tTags: {}'.format(t2i.keys()))

        print('\tmaximum length of sentence: {}'.format(msent))
        print('\tNumber of sentences: {}'.format(lines))
        print('\n')
        info.write('\n')
        info.write('Maximum length of sentence in dataset: {}\n'.format(msent))
        info.write('Total quantity of sentences in dataset: {}\n'.format(lines))


# TODO: make json file with info about number of tokens and anover informationm like maximum lengh of the sentence
# TODO: t2i and i2t dicts

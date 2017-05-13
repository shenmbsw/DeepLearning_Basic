#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import copy
import re

import six
import numpy as np

tmp_path = '/tmp/'
url = ('https://raw.githubusercontent.com/fluentpython/example-code/master/attic/'
       'sequences/war-and-peace.txt')


class GeneratorRestartHandler(object):
    def __init__(self, gen_func, argv, kwargv):
        self.gen_func = gen_func
        self.argv = copy.copy(argv)
        self.kwargv = copy.copy(kwargv)
        self.local_copy = self.gen_func(*self.argv, **self.kwargv)

    def __iter__(self):
        return GeneratorRestartHandler(self.gen_func, self.argv, self.kwargv)

    def __next__(self):
        return next(self.local_copy)


def restartable(g_func):
    def tmp(*argv, **kwargv):
        return GeneratorRestartHandler(g_func, argv, kwargv)

    return tmp


def _read_dataset_war_and_peace():
    file_path = os.path.join(tmp_path, 'w_and_p.txt')
    if not os.path.exists(file_path):
        six.moves.urllib.request.urlretrieve(url, file_path)

    with open(file_path, 'r') as f:
        file_content = f.read()

    file_content = re.sub(' +', ' ', file_content.replace('\n', ' '))
    used_chars = list(set(file_content))
    char2int_dict = dict(zip(used_chars, range(len(used_chars))))
    int2char_dict = dict(zip(range(len(used_chars)), used_chars))
    id_seq = [char2int_dict[char] for char in file_content]
    return id_seq, char2int_dict, int2char_dict


@restartable
def _batch_tolstoy_generator(id_seq, batch_size=200, seq_size=100):
    n_batches = int(len(id_seq)/batch_size)
    for i in range(n_batches):
        batch_list = []
        for j in range(batch_size):
            start_idx = random.randint(1, len(id_seq)-seq_size-5)
            batch_x = id_seq[start_idx:start_idx+seq_size]
            batch_y = id_seq[start_idx+seq_size]
            batch_list.append((batch_x, batch_y))
        batch_x_list, batch_y_list = zip(*batch_list)
        yield np.array(batch_x_list), np.array(batch_y_list)


def batch_tolstoy_generator(*arvg, **kwargs):
    id_seq, char2int_dict, int2char_dict = _read_dataset_war_and_peace()
    gen = _batch_tolstoy_generator(id_seq, *arvg, **kwargs)
    return gen, char2int_dict, int2char_dict


def _print_data_specs():
    seq, map_dict, backmap_dict = _read_dataset_war_and_peace()
    print(len(seq))

    btg, char2int_dict, int2char_dict = batch_tolstoy_generator(10, 5)
    X, y = next(btg)
    print(X.shape, y.shape)
    print(X[:5], y[:5])
    list(btg)

    X, y = next(iter(btg))
    print(X.shape, y.shape)
    print(X[:5], y[:5])

if __name__ == '__main__':
    _print_data_specs()

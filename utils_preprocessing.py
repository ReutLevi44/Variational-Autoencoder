import os
import shutil
import numpy as np
import pandas as pd
from shutil import copy2


# cdr3s to one hot vectors
def data_preprocessing(string_set, max_length, is_default=False):
    # one hot vector per amino acid dictionary- wih STOP sequence- !
    aa = ['V', 'I', 'L', 'E', 'Q', 'D', 'N', 'H', 'W', 'F',
          'Y', 'R', 'K', 'S', 'T', 'M', 'A', 'G', 'P', 'C', '!']
    n_aa = len(aa)
    one_hot = {a: [0] * n_aa for a in aa}
    for key in one_hot:
        one_hot[key][aa.index(key)] = 1
    # add zero key for the zero padding
    one_hot['0'] = [0] * n_aa
    # add 1 to the maximum length ( +1 for the ! stop signal)
    max_length += 1
    # generate one-hot long vector for each cdr3
    one_vecs = []
    for ind, cdr3 in enumerate(string_set):
        # in default, the cdr3 cut to 24 = max len in default autoencoder
        if is_default:
            if len(cdr3) > 24:
                cdr3 = cdr3[:24]
        # add stop signal in each sequence
        cdr3 = cdr3 + '!'
        my_len = len(cdr3)
        # zero padding in the end of the sequence
        if is_default:
            if my_len < 25:
                add = 25 - my_len
                cdr3 = cdr3 + '0' * add
        else:
            if my_len < max_length:
                add = max_length - my_len
                cdr3 = cdr3 + '0' * add
        # one hot vectors
        v = []
        for c in cdr3:
            v += one_hot[c]
        one_vecs.append(v)
    return one_vecs


# cdr3s to one hot vectors
def data_preprocessing_with_v(string_set, vs_set, max_length, n_vs):
    # one hot vector per amino acid dictionary- wih STOP sequence- !
    aa = ['V', 'I', 'L', 'E', 'Q', 'D', 'N', 'H', 'W', 'F',
          'Y', 'R', 'K', 'S', 'T', 'M', 'A', 'G', 'P', 'C', '!']
    n_aa = len(aa)
    one_hot = {a: [0] * n_aa for a in aa}
    for key in one_hot:
        one_hot[key][aa.index(key)] = 1
    # add zero key for the zero padding
    one_hot['0'] = [0] * n_aa
    # v one-hot vectors
    all_vs = list(set(vs_set))
    one_hot_v = {v: [0] * n_vs for v in all_vs}
    for v in one_hot_v:
        one_hot_v[v][all_vs.index(v)] = 1
    # add 1 to the maximum length ( +1 for the ! stop signal)
    max_length += 1
    # generate one-hot long vector for each cdr3
    one_vecs = []
    for ind, cdr3 in enumerate(string_set):
        # add stop signal in each sequence
        cdr3 = cdr3 + '!'
        my_len = len(cdr3)
        # zero padding in the end of the sequence
        if my_len < max_length:
            add = max_length - my_len
            cdr3 = cdr3 + '0' * add
        # one hot vectors
        v = []
        for c in cdr3:
            v += one_hot[c]
        v += one_hot_v[vs_set[ind]]
        one_vecs.append(v)
    return one_vecs


# predictions to one-hot vectors
def hardmax_zero_padding(l):
    n = 21
    l_chunks = [l[i:i + n] for i in range(0, len(l), n)]
    l_new = []
    for chunk in l_chunks:
        new_chunk = list(np.zeros(n, dtype=int))
        # # taking the max only in place where not everything is 0
        # if not all(v == 0 for v in chunk):
        max = np.argmax(chunk)
        # if max == 20:
        #     break
        new_chunk[max] = 1
        l_new += new_chunk
    return l_new


# count mismatches between an input vector and the predicted vector
def count_mismatches_zero_padding(a, b):
    n = 21
    a_chunks = [a[i:i + n] for i in range(0, len(a), n)]
    b_chunks = [b[i:i + n] for i in range(0, len(b), n)]
    count_err = 0
    for ind, chunck_a in enumerate(a_chunks):
        ind_a = ''.join(str(x) for x in chunck_a).find('1')
        ind_b = ''.join(str(x) for x in b_chunks[ind]).find('1')
        if ind_a != ind_b:
            count_err += 1
        # early stopping when there are allready 2 mismatches
        if count_err > 2:
            return 3
    return count_err


# check vadility of CDR3 sequence
def check_cdr3(seq):
    chars = ['#', 'X', '*', '_']
    if seq == '':
        return False
    for c in chars:
        if c in seq:
            return False
    return True


# find the tag of a sample according to the dataset, add datasets if needed
def tag_per_data_set(file_name, root):
    my_tag = 'error'
    if 'vaccine' in root:
        tmp = file_name.split('.')[0]
        my_tag = tmp.split('^')[1]
    if 'cancer' in root:
        tmp = file_name[0]
        if tmp == 'c':
            my_tag = 'Cancer'
        else:
            my_tag = 'Healthy'
    if 'benny_chain' in root:
        if 'naive' in file_name:
            my_tag = 'Naive'
        else:
            my_tag = 'Memory'
    if 'Glanville' in root:
        my_tag = file_name.split('.')[0]
    if 'Rudqvist' in root:
        my_tag = file_name.split('-')[0]
    if 'Sidhom' in root:
        if 'SIY' in file_name:
            my_tag = 'SIY'
        else:
            my_tag = 'TRP2'
    return my_tag


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)


def dict_to_df(d, x_str, y_str):
    x = []
    y = []
    ordered_keys = np.sort(list(d.keys()))
    for key in ordered_keys:
        x += [key] * len(d[key])
        y += list(d[key])
    return pd.DataFrame(data={x_str: x, y_str: y})


def create_dir_default(root, file_name):
    if not os.path.exists(r'./weights_' + root):
        os.mkdir(r'./weights_' + root)
    copy2('./weights_Default/' + file_name + '.h5', './weights_' + root)

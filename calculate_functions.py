import numpy as np
import csv
import random
import jellyfish
from sklearn.neighbors import KernelDensity
import scipy.spatial.distance as dist
from sklearn import metrics

from utils_preprocessing import hardmax_zero_padding, count_mismatches_zero_padding


# calculate accuracy, all vectors
def calc_accuracy_zero_padding(inputs, y, path):
    acc = 0
    acc1 = 0
    acc2 = 0
    n = len(inputs)
    for i in range(n):
        hard_max_y = hardmax_zero_padding(y[i])
        real = list(inputs[i])
        hard_max_y = cut_by_stop(hard_max_y)
        real = cut_by_stop(real)
        if len(hard_max_y) != len(real):
            continue
        # cut the output to be the same length as input
        # real = real[:len(hard_max_y)]
        if real == hard_max_y:
            acc += 1
            acc1 += 1
            acc2 += 1
        else:
            # accept 1 mismatch aa
            err = count_mismatches_zero_padding(real, hard_max_y)
            if err == 1:
                acc1 += 1
                acc2 += 1
            else:
                # accept 2 mismatch aa
                if err == 2:
                    acc2 += 1
    print('accuracy: ' + str(acc) + '/' + str(n) + ', ' + str(np.round((acc / n) * 100, 2)) + '%')
    print('1 mismatch accuracy: ' + str(acc1) + '/' + str(n) + ', ' + str(np.round((acc1 / n) * 100, 2)) + '%')
    print('2 mismatch accuracy: ' + str(acc2) + '/' + str(n) + ', ' + str(np.round((acc2 / n) * 100, 2)) + '%')
    with open(path + 'autoencoder_accuracy.csv', 'w') as csvfile:
        fieldnames = ['Accuracy', '1 Mismatch', '2 Mismatch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy': str(np.round((acc / n) * 100, 2)), '1 Mismatch': str(np.round((acc1 / n) * 100, 2)),
                         '2 Mismatch': str(np.round((acc2 / n) * 100, 2))})


# kernel density distance
def kde_distance(set1, set2, flag=False):
    # calc distance matrix between both sets
    h = 1.06 * np.std(set1) * len(set1) ** (-1 / 5)
    kde = KernelDensity(bandwidth=h, kernel='gaussian')
    kde.fit(set2)
    dis = kde.score_samples(set1)
    # dont check density of point with itself when fit and score are on the same set-
    # remove constant K(0)/h from each point in the array
    if flag:
        zeros_set = [np.zeros(set1.shape[1])]
        kde_const = KernelDensity(bandwidth=h, kernel='gaussian')
        kde_const.fit(zeros_set)
        const = kde_const.score_samples(zeros_set)
        dis += const
    return np.average(np.exp(dis))


# find the center of all projections
def find_center(data):
    vecs = [v[0] for v in data]
    CM = np.average(vecs, axis=0)
    return CM


# find the all distanced from center
def distances_from_center(data, cm, public):
    distances = dict()
    vecs = [i[0] for i in data]
    dis_mat = dist.cdist(vecs, [cm])
    for j, p in enumerate(data):
        d = dis_mat[j][0]
        n = int(public[p[1]])
        distances[n] = distances.get(n, []) + [d]
    return distances


# KDE distance between every two samples
def compute_kde_distances(vecs_dict_):
    dis_dict_ = dict()
    for i, file in enumerate(vecs_dict_):
        dis_dict_[file] = dict()
        for j, file2 in enumerate(vecs_dict_):
            # non-simetric distance metric
            if i == j:
                # keep seperate dict for self distances for bar plot
                dis_dict_[file][file2] = kde_distance(vecs_dict_[file], vecs_dict_[file2], flag=True)
            else:
                dis_dict_[file][file2] = kde_distance(vecs_dict_[file], vecs_dict_[file2])

            # for simetric distance metric
            # if i < j:
            #     dis_dict_[file][file2] = kde_distance(vecs_dict_[file], vecs_dict_[file2])
            # else:
            #     dis_dict_[file][file2] = dis_dict_[file2][file]

    return dis_dict_


# calculate V distribution of a sample
def calc_distribution(vs, vs_set):
    vs_dict = dict()
    # init with low values instead of 0 for the KL
    for v in vs_set:
        vs_dict[v] = 0.1
    # count segments
    for v in vs:
        vs_dict[v] += 1
    # find distribution
    n = len(vs)
    for v in vs_dict:
        vs_dict[v] = vs_dict[v] / n
    dist = []
    for v in vs_set:
        dist.append(vs_dict[v])
    return dist


# calculate KL between two distributions
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


# final distance between two samples
def similarity_min_avg(set1, set2, flag=False):
    d_list = []
    for ind1, seq1 in enumerate(set1):
        tmp = []
        for ind2, seq2 in enumerate(set2):
            if flag and ind1 == ind2:
                continue
            tmp.append(jellyfish.levenshtein_distance(seq1, seq2))
        d_list.append(min(tmp))
    return np.average(d_list)


# compute similarity between each pair of compartments using func
# for efficiency- only upper half will be computed and the results
# .. are saved for later lowe half
def cxc_similarity(d):
    data = []
    my_order = []
    seen = dict()
    for ind1, c1 in enumerate(d):
        my_order.append(c1)
        r = []
        for ind2, c2 in enumerate(d):
            sample1 = d[c1]
            sample2 = d[c2]
            if ind1 < ind2:
                # a measure of how similar both samples are according their cdr3 sets
                val = similarity_min_avg(sample1, sample2)
                seen[c2 + 'x' + c1] = val
            elif ind1 == ind2:
                # a measure of how similar both samples are according their cdr3 sets
                val = similarity_min_avg(sample1, sample2, flag=True)
                seen[c2 + 'x' + c1] = val
            else:
                val = seen[c1 + 'x' + c2]
            r.append(val)
        data.append(r)
    return data, my_order


# KDE within a sample
def self_density(d, public):
    densities = dict()
    for s in d:
        sample_vecs = [i[0] for i in d[s]]
        for i, val in enumerate(d[s]):
            kde = KernelDensity(bandwidth=3.0, kernel='gaussian')
            kde.fit(sample_vecs[:i] + sample_vecs[i + 1:])
            dis = kde.score_samples([val[0]])
            density = np.exp(dis[0])
            n = int(public[val[1]])
            densities[n] = densities.get(n, []) + [density]
    return densities


# aux function to calc_accuracy_zero_padding
def cut_by_stop(l):
    n = 21
    new_chunck = []
    l_chunks = [l[i:i + n] for i in range(0, len(l), n)]
    for ind, chunk in enumerate(l_chunks):
        new_chunck += chunk
        max = np.argmax(chunk)
        if max == 20:
            return new_chunck
    return []

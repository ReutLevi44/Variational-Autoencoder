"""
This module contain all loading function
"""
import csv
import random
import pickle

from utils_preprocessing import *
from colorama import Fore
from tqdm import tqdm


# load data by path - n data
def load_n_data(path, p, del_str, aa_str, v_str, f_str):
    all_data = []
    all_vs = []
    n_data = []
    n_tags = []
    n_files = []
    n_freqs = []
    n_vs = []
    count_rows = 0
    breaker = False
    breaker2 = False
    for directory, subdirectories, files in os.walk(path):
        for i, file in tqdm(enumerate(files), desc="Load File", bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                data = []
                vs = []
                freqs = []
                for row in reader:
                    if count_rows == 1000000:
                        print("\nThe data was too big and therefore cut short")
                        breaker = True
                        break
                    cdr3 = row[aa_str]
                    if not check_cdr3(cdr3):
                        continue
                    data.append(cdr3)
                    if f_str != 'None':
                        freqs.append(row[f_str])
                    if v_str != 'None':
                        vs.append(row[v_str])
                    count_rows += 1
                all_data += data
                all_vs += vs

                # sample data by p
                if isinstance(p, float):
                    sample_n = int(len(data) * p)
                else:
                    sample_n = p
                inds = random.sample(list(range(len(data))), sample_n)
                n_data += [data[i] for i in inds]

                # freqs to counts
                if f_str != 'None':
                    n_freqs += [(float(freqs[i])) for i in inds]
                if v_str != 'None':
                    n_vs += [vs[i] for i in inds]

                # # tags
                # t = tag_per_data_set(file, path)
                # if t == 'error':
                #     print('Warning: Add Data Set To (tag_per_data_set) Function !')
                #     quit()
                # n_tags += [t] * sample_n

                n_files += [file] * sample_n
                if breaker:
                    breaker2 = True
                    break
        if breaker2:
            break
    # check maximal length
    max_length = np.max([len(s) for s in all_data])
    return n_data, n_vs, n_tags, n_files, n_freqs, max_length, len(set(all_vs))


# load CDR3 sequences per sample
def load_seqs_by_samples(r, del_str, aa_str, n):
    seqs_dict = dict()
    for d, subd, files in os.walk(r):
        for file in files:
            with open(os.path.join(d, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                data = []
                for row in reader:
                    cdr3 = row[aa_str]
                    if not check_cdr3(cdr3):
                        continue
                    data.append(cdr3)
                if len(data) < n:
                    print(file + ': smaller then ' + str(n))
                    continue
                seqs_dict[file.split('.')[0]] = np.random.choice(data, n)
    return seqs_dict


# load V genes per sample
def load_vs_by_samples(r, del_str, v_str, n):
    all_vs = []
    vs_dict = dict()
    for d, subd, files in os.walk(r):
        for file in files:
            with open(os.path.join(d, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                vs = [row[v_str] for row in reader]
                if len(vs) < n:
                    print(file + ': smaller then ' + str(n))
                    continue
                vs = list(np.random.choice(vs, n))
                all_vs += vs
                vs_dict[file.split('.')[0]] = vs
    return vs_dict, set(all_vs)


# load all of type data- given length n, returns a dictionary of processed data per file
def load_all_data(path, n, del_str, aa_str, f_str, is_default=False):
    all_data = {}
    with open(path, mode='r') as infile:
        reader = csv.DictReader(infile, delimiter=del_str)
        seqs = []
        freqs = []
        for row in reader:
            cdr3 = row[aa_str]
            if not check_cdr3(cdr3):
                continue
            seqs.append(cdr3)
            if f_str != 'None':
                freqs.append(int(float(row[f_str])))
        vecs = data_preprocessing(seqs, n, is_default)
        all_data[path.split('.')[0].split('/')[1]] = {'vecs': vecs, 'seqs': seqs, 'freqs': freqs}
    return all_data


# load all of type data- given length n, returns a dictionary of processed data per file
def load_all_data_with_v(path, n, del_str, aa_str, v_str, vs_n, f_str):
    all_data = {}
    with open(path, mode='r') as infile:
        reader = csv.DictReader(infile, delimiter=del_str)
        seqs = []
        vs = []
        freqs = []
        for row in reader:
            cdr3 = row[aa_str]
            if not check_cdr3(cdr3):
                continue
            seqs.append(cdr3)
            if v_str != 'None':
                vs.append(row[v_str])
            if f_str != 'None':
                freqs.append(int(float(row[f_str])))
        vecs = data_preprocessing_with_v(seqs, vs, n, vs_n)
        all_data[path.split('.')[0].split('/')[1]] = {'vecs': vecs, 'seqs': seqs, 'freqs': freqs}
    return all_data


# load projections by sample
def load_representations(path, n):
    samples = dict()
    for file in os.listdir(path):
        if len(file.split('.')) == 1 or file.split('.')[1] != 'p':
            continue
        vecs_array = pickle.load(open(os.path.join(path, file), "rb"))
        if len(vecs_array) < n:
            # n = len(vecs_array)
            vecs_array = random.sample([vec[0] for vec in vecs_array], len(vecs_array))
        else:
            vecs_array = random.sample([vec[0] for vec in vecs_array], n)
        samples[file] = np.array(vecs_array)
    return samples


# load projections by frequencies by sample
def load_representations_by_freqs(path, n):
    samples = dict()
    for file in os.listdir(path):
        if len(file.split('.')) == 1 or file.split('.')[1] != 'p':
            continue
        vecs_array = pickle.load(open(os.path.join(path, file), "rb"))
        freq_lst = []
        if vecs_array[0][2] < 1:
            for v_f in vecs_array:
                freq_lst.append(v_f[2])
            min_freq = min(freq_lst)
        if len(vecs_array) < n:
            # n = len(vecs_array)
            vecs_array = random.sample([[vec[0], vec[2]] for vec in vecs_array], len(vecs_array))
        else:
            vecs_array = random.sample([[vec[0], vec[2]] for vec in vecs_array], n)
        vecs_array_fs = []
        for v in vecs_array:
            # for the frequency (the calculate doesnt work yet)
            # if v[1] < 1:
            #     v[1] = round(v[1] / min_freq)
            v[1] = int(v[1])
            vecs_array_fs += [list(v[0])] * v[1]
        samples[file] = np.array(vecs_array_fs)
    return samples


# load projections with sequences by sample
def load_representations_and_seqs(path, n):
    samples = dict()
    for file in os.listdir(path):
        if len(file.split('.')) == 1 or file.split('.')[1] != 'p':
            continue
        vecs_array = pickle.load(open(os.path.join(path, file), "rb"))
        if len(vecs_array) < n:
            # n = len(vecs_array)
            vecs_array = random.sample([[vec[0], vec[1]] for vec in vecs_array], len(vecs_array))
        else:
            vecs_array = random.sample([[vec[0], vec[1]] for vec in vecs_array], n)
        samples[file] = vecs_array
    return samples


# find publicity of each CDR3 sequence
def public_cdr3s(path, del_str, aa_str):
    seqs = {}
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                for row in reader:
                    cdr3 = row[aa_str]
                    if not check_cdr3(cdr3):
                        continue
                    if cdr3 not in seqs:
                        seqs[cdr3] = set()
                    seqs[cdr3].add(file.split('.')[0])
    seqs = {seq: len(val) for seq, val in seqs.items()}
    return seqs


# save KDE matrix to csv file
def kde_to_csv(dis_dict_, headers_, path_to_csv_):
    # fill the full matrix of the data
    with open(path_to_csv_, 'w') as csvfile:
        fieldnames = ['file'] + headers_
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, f in enumerate(headers_):
            data = {'file': f}
            for j, f2 in enumerate(headers_):
                # keep the real value in the csv file
                data[f2] = dis_dict_[f][f2]
            writer.writerow(data)


# read csv file to array
def csv_to_arr(path_to_csv_):
    headers_ = []
    data_arr_ = []
    self_arr_ = []
    with open(path_to_csv_, mode='r') as infile:
        reader = csv.DictReader(infile)
        for i, row in enumerate(reader):
            del row['file']
            if i == 0:
                headers_ = list(row.keys())
            tmp = []
            for j, key in enumerate(row):
                val = np.float(row[key])
                if i == j:
                    self_arr_.append(val)
                    tmp.append(np.nan)
                else:
                    tmp.append(val)
            data_arr_.append(tmp)
    return data_arr_, self_arr_, headers_


def remove_pickle_files(path):
    for d, dsub, files in os.walk(path):
        for file in files:
            if len(file.split('.')) != 1 and file.split('.')[1] == 'p':
                os.remove(d + "/" + file)

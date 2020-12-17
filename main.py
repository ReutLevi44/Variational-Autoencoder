import json
import csv

import scipy
from keras import Sequential
from keras.engine import Layer
from scipy import spatial
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.metrics import mean_squared_error

import tensorflow.keras.backend as K
from sympy import Add
from tensorflow.keras.models import load_model

# import all customer modules that included class
# from classifier import Classifier

from v_gens_variational_auto_encoder import VariationalVAutoEncoder
from v_gens_variational_embedding import VEmbeddingVariationalAutoEncoder

from embedding_variational_auto_encoder import EmbeddingVariationalAutoEncoder
from variational_autoencoder import VariationalAutoEncoder

# import all service functions
from utils_preprocessing import *
from initialize_files import *
from plot_functions import *
from calculate_functions import *
from devidedata2train_test import *


# check correct of  all module
if __name__ == '__main__':

    # Load Data ----------------------------------------------

    print('loading data')

    with open('parameters.json') as f:
        parameters = json.load(f)

    root = parameters["root"]  # path to data set
    save_path = root + '_Results'

    # replace the headers with those in the 'csv' file, put 'None' if missing
    DEL_S = parameters["DEL_S"]  # which delimiter to use to separate headers
    CDR3_S = parameters["CDR3_S"]  # header of amino acid sequence of CDR3
    V_S = parameters["V_S"]  # header of V gene, 'None' for data set with no resolved V
    F_S = parameters["F_S"]  # header of clone frequency, 'None' for data set with no frequency

    EPOCHS = parameters["EPOCHS"]  # number of epochs for each model
    P_LOAD = parameters["P_LOAD"]  # number of data to load for training, float or int
    ENCODING_DIM = parameters["ENCODING_DIM"]  # number of dimensions in the embedded space
    N_PROJECTIONS = parameters["N_PROJECTIONS"]  # number of projections to use in further anlysis
    N_FOR_DISTANCES = parameters["N_FOR_DISTANCES"]  # number of projections to use in the ED and KL

    # each directory contains the projections for each sample and all the results
    encoder_dir = os.path.join(save_path, parameters["encoder_dir"] + "/")
    v_encoder_dir = os.path.join(save_path, parameters["v_encoder_dir"] + "/")
    embedding_dir = os.path.join(save_path, parameters["embedding_dir"] + "/")
    v_embedding_dir = os.path.join(save_path, parameters["v_embedding_dir"] + "/")

    # each directory contains the projection by csv format to user
    encoder_dir_to_user = os.path.join(encoder_dir, 'projections_to_user' + "/")
    v_encoder_dir_to_user = os.path.join(v_encoder_dir, 'projections_to_user' + "/")
    embedding_dir_to_user = os.path.join(embedding_dir, 'projections_to_user' + "/")
    v_embedding_dir_to_user = os.path.join(v_embedding_dir, 'projections_to_user' + "/")

    # Run Parameters- change the analysis you wish to run to True, otherwise- False

    # several encoders
    RUN_AE = parameters["RUN_AE"]
    RUN_V_AE = parameters["RUN_V_AE"]
    RUN_EMB = parameters["RUN_EMB"]
    RUN_V_EMB = parameters["RUN_V_EMB"]

    # default encoders
    RUN_AE_DEFAULT = parameters["RUN_AE_DEFAULT"]
    RUN_EMB_DEFAULT = parameters["RUN_EMB_DEFAULT"]

    # configuration of logger in DNN
    RUN_SAVE = parameters["RUN_SAVE"]
    RUN_TSNE = parameters["RUN_TSNE"]
    RUN_KDE = parameters["RUN_KDE"]
    RUN_PROPS = parameters["RUN_PROPS"]
    RUN_KL = parameters["RUN_KL"]
    RUN_ED = parameters["RUN_ED"]
    RUN_MDS = parameters["RUN_MDS"]

    # load data
    data, vs, all_tags, files, freqs, MAX_LEN, V_ONE_HOT_LEN = load_n_data(root, P_LOAD, DEL_S, CDR3_S, V_S, F_S)
    # for emb, cut the len for the calculate
    data_emb = data
    if len(data) > 10000:
        data_emb = data[:10000]
    vecs_data = data_preprocessing(data, MAX_LEN, RUN_AE_DEFAULT or RUN_EMB_DEFAULT)  # representation of the sequences as one-hot vectors
    vecs_data_emb = data_preprocessing(data_emb, MAX_LEN, RUN_AE_DEFAULT or RUN_EMB_DEFAULT)

    # if the dataset contains V, another representation of the sequences as one-hot vectors with V
    if V_S != 'None':
        vecs_data_vs = data_preprocessing_with_v(data, vs, MAX_LEN, V_ONE_HOT_LEN)
        vecs_data_vs_emb = data_preprocessing_with_v(data_emb, vs, MAX_LEN, V_ONE_HOT_LEN)

    # one color for each category within the dataset, add if necessary
    colors = ['gold', 'm', 'blue', 'lime', 'orchid', 'grey', 'r']

    if RUN_AE_DEFAULT:
        create_dir_default(root, "encoder_weights")

    if RUN_EMB_DEFAULT:
        create_dir_default(root, "embedding_encoder_weights")

    # Variational Auto-encoder ----------------------------------------------
    # run the basic auto-encoder

    if RUN_AE or RUN_AE_DEFAULT:
        print('Variational Autoencoder')

        create_dir(encoder_dir)
        create_dir(encoder_dir_to_user)

        if RUN_AE:
            # train + test sets
            train_X, test_X, tmp1, tmp2 = train_test_split(vecs_data, vecs_data, test_size=0.20)

            v_ae = VariationalAutoEncoder(np.array(train_X), root, encoding_dim=ENCODING_DIM)
            vae_model = v_ae.v_encoder_decoder()

            train_results = v_ae.fit_v_autoencoder(np.array(train_X), batch_size=50, epochs=EPOCHS)
            v_ae.save_vae()

            # plot loss and accuracy as a function of time
            plot_loss(train_results.history['loss'], train_results.history['val_loss'], encoder_dir)

            test_X = np.array(test_X)
            train_X = np.array(train_X)

            y = vae_model.predict(test_X)

            # accuracy
            calc_accuracy_zero_padding(test_X, y, encoder_dir)

            # # Calculate the test MSE
            #
            # predict_test = vae_model.predict(test_X)
            # predict_train = vae_model.predict(train_X)
            #
            # print(f'\nThe MSE of the train is: {mean_squared_error(train_X, predict_train[0])}')
            # print(f'\nThe MSE of the test is: {mean_squared_error(test_X, predict_test[0])}')

    # Variational Auto-encoder With V ----------------------------------------------
    # run the auto-encoder with V

    if RUN_V_AE and V_S != 'None':
        print('v variational-auto-encoder')

        # train + test sets
        train_X, test_X, tmp1, tmp2 = train_test_split(vecs_data_vs, vecs_data_vs, test_size=0.20)

        # train  auto-encoder model
        v_vae = VariationalVAutoEncoder(np.array(train_X), root, V_ONE_HOT_LEN, encoding_dim=ENCODING_DIM)
        v_vae_model = v_vae.v_encoder_decoder()
        train_results = v_vae.fit_v_autoencoder(np.array(train_X), batch_size=50, epochs=EPOCHS)
        v_vae.save_vae()

        create_dir(v_encoder_dir)
        create_dir(v_encoder_dir_to_user)

        # plot loss and accuracy as a function of time
        plot_loss(train_results.history['loss'], train_results.history['val_loss'], v_encoder_dir)

        # test model
        test_X = np.array(test_X)
        train_X = np.array(train_X)

        y = v_vae_model.predict(test_X)

        # accuracy
        calc_accuracy_zero_padding(test_X, y, v_encoder_dir)

    # Embedding Variational Auto-encoder ----------------------------------------------
    # run the auto-encoder with respect to the original distances

    if RUN_EMB or RUN_EMB_DEFAULT:
        print('embedding')

        create_dir(embedding_dir)
        create_dir(embedding_dir_to_user)

        if RUN_EMB:
            # train + test sets
            train_X, test_X, train_y, test_y = train_test_split(vecs_data_emb, vecs_data_emb, test_size=0.2)
            train_X = np.array(train_X)

            # calculate D- original distances matrix- norm 2 between all pairs of one-hot vectors
            D = spatial.distance.cdist(train_X, train_X, 'euclidean')

            # train model
            emb_vae = EmbeddingVariationalAutoEncoder(train_X, D, root, encoding_dim=ENCODING_DIM, batch_size=50,
                                                      emb_alpha=0.000001)

            emb_vae_model = emb_vae.v_encoder_decoder()
            embedding_train = emb_vae.fit_generator(epochs=EPOCHS)
            emb_vae.save_vae()

            create_dir(embedding_dir)
            create_dir(embedding_dir_to_user)

            # test model
            test_X = np.array(test_X)
            train_X = np.array(train_X)

            y = emb_vae_model.predict(test_X)[0]

            # accuracy
            calc_accuracy_zero_padding(test_X, y, embedding_dir)

    # Embedding Auto-encoder With V ----------------------------------------------
    # run the auto-encoder with respect to the original distances + V representation

    if RUN_V_EMB and V_S != 'None':
        print('v embedding')

        # train + test sets
        train_X, test_X, train_y, test_y = train_test_split(vecs_data_vs_emb, vecs_data_vs_emb, test_size=0.2)
        train_X = np.array(train_X)

        # calculate D- input distances matrix- norm 2
        D = spatial.distance.cdist(train_X, train_X, 'euclidean')

        # train model
        v_emb_vae = VEmbeddingVariationalAutoEncoder(train_X, D, root, V_ONE_HOT_LEN, encoding_dim=ENCODING_DIM,
                                                     batch_size=50, emb_alpha=0.000001)

        v_emb_vae_model = v_emb_vae.v_encoder_decoder()
        v_embedding_train = v_emb_vae.fit_generator(epochs=EPOCHS)
        v_emb_vae.save_vae()

        create_dir(v_embedding_dir)
        create_dir(v_embedding_dir_to_user)

        test_X = np.array(test_X)
        train_X = np.array(train_X)

        y = v_emb_vae_model.predict(test_X)[0]

        # accuracy
        calc_accuracy_zero_padding(test_X, y, v_embedding_dir)

    # Save Projections ----------------------------------------------
    # save all projections per sample for each of the trained models

    if RUN_SAVE:

        print('save projections')
        for directory, subdirectories, files in os.walk(root):
            for file in files:
                # load data (without V) and models
                samples = load_all_data(directory + '/' + file, MAX_LEN, DEL_S, CDR3_S, F_S,
                                        RUN_AE_DEFAULT or RUN_EMB_DEFAULT)
                if RUN_AE or RUN_AE_DEFAULT:
                    encoder = load_model(r'./weights_' + root + '/encoder_weights.h5')
                if RUN_EMB or RUN_EMB_DEFAULT:
                    embedding_encoder = load_model(r'./weights_' + root + '/embedding_encoder_weights.h5')

                # save all projections per sample
                for s in samples:

                    if RUN_AE or RUN_AE_DEFAULT:
                        # encoder projections
                        x_encoder = encoder.predict(np.array(samples[s]['vecs']))
                        if F_S != 'None':
                            info = [[x_encoder[i], samples[s]['seqs'][i], samples[s]['freqs'][i]] for i in range(len(x_encoder))]
                        else:
                            info = [[x_encoder[i], samples[s]['seqs'][i]] for i in range(len(x_encoder))]

                        # if RUN_AE:
                        with open(encoder_dir_to_user + str(s) + '.csv', mode='w') as output_file:
                            output_writer = csv.writer(output_file, delimiter=',')
                            for item in info:
                                if F_S != 'None':
                                    vector, name, _ = item
                                else:
                                    vector, name = item
                                row = list(map(lambda n: '%.8f'%n, vector))
                                row.insert(0, name)
                                output_writer.writerow(row)

                        pickle.dump(info, open(encoder_dir + str(s) + '.p', "wb"))

                    if RUN_EMB or RUN_EMB_DEFAULT:
                        # encoder+distances projections
                        x_embedding = embedding_encoder.predict(np.array(samples[s]['vecs']))
                        if F_S != 'None':
                            info = [[x_embedding[i], samples[s]['seqs'][i], samples[s]['freqs'][i]] for i in
                                    range(len(x_embedding))]
                        else:
                            info = [[x_embedding[i], samples[s]['seqs'][i]] for i in range(len(x_embedding))]

                        # if RUN_EMB:
                        with open(embedding_dir_to_user + str(s) + '.csv', mode='w') as output_file:
                            output_writer = csv.writer(output_file, delimiter=',')
                            for item in info:
                                if F_S != 'None':
                                    vector, name, _ = item
                                else:
                                    vector, name = item
                                row = list(map(lambda n: '%.8f' % n, vector))
                                row.insert(0, name)
                                output_writer.writerow(row)

                        pickle.dump(info, open(embedding_dir + str(s) + '.p', "wb"))

                if V_S != 'None':

                    # load data with V and models
                    samples_with_v = load_all_data_with_v(directory + '/' + file, MAX_LEN, DEL_S, CDR3_S, V_S,
                                                          V_ONE_HOT_LEN, F_S)
                    if RUN_V_AE:
                        v_encoder = load_model(r'./weights_' + root + '/v_encoder_weights.h5')
                    if RUN_V_EMB:
                        v_embedding_encoder = load_model(r'./weights_' + root + '/v_embedding_encoder_weights.h5')

                    # save all projections per sample
                    for s in samples_with_v:

                        if RUN_V_AE:
                            # encoder projections
                            x_v_encoder = v_encoder.predict(np.array(samples_with_v[s]['vecs']))
                            if F_S != 'None':
                                info = [[x_v_encoder[i], samples_with_v[s]['seqs'][i], samples_with_v[s]['freqs'][i]] for i in
                                        range(len(x_v_encoder))]
                            else:
                                info = [[x_v_encoder[i], samples_with_v[s]['seqs'][i]] for i in range(len(x_v_encoder))]

                            # if RUN_V_AE:
                            with open(v_encoder_dir_to_user + str(s) + '.csv', mode='w') as output_file:
                                output_writer = csv.writer(output_file, delimiter=',')
                                for item in info:
                                    if F_S != 'None':
                                        vector, name, _ = item
                                    else:
                                        vector, name = item
                                    row = list(map(lambda n: '%.8f' % n, vector))
                                    row.insert(0, name)
                                    output_writer.writerow(row)

                            pickle.dump(info, open(v_encoder_dir + str(s) + '.p', "wb"))

                        if RUN_V_EMB:
                            # encoder+distances projections
                            x_v_embedding_encoder = v_embedding_encoder.predict(np.array(samples_with_v[s]['vecs']))
                            if F_S != 'None':
                                info = [[x_v_embedding_encoder[i], samples_with_v[s]['seqs'][i], samples_with_v[s]['freqs'][i]] for
                                        i in
                                        range(len(x_v_embedding_encoder))]
                            else:
                                info = [[x_v_embedding_encoder[i], samples_with_v[s]['seqs'][i]] for i in
                                        range(len(x_v_embedding_encoder))]

                            # if RUN_V_EMB:
                            with open(v_embedding_dir_to_user + str(s) + '.csv', mode='w') as output_file:
                                output_writer = csv.writer(output_file, delimiter=',')
                                for item in info:
                                    if F_S != 'None':
                                        vector, name, _ = item
                                    else:
                                        vector, name = item
                                    row = list(map(lambda n: '%.8f' % n, vector))
                                    row.insert(0, name)
                                    output_writer.writerow(row)

                            pickle.dump(info, open(v_embedding_dir + str(s) + '.p', "wb"))

    # TSNE ----------------------------------------------
    # visualize all samples using TSNE

    if RUN_TSNE:

        print('TSNE')

        # projections to use
        path = os.path.join(save_path, parameters["tsne_props_encoder"] + "/")

        # load projections
        vecs_dict = load_representations(path, N_PROJECTIONS)

        # calculte TSNE
        inds = []
        for i, f in enumerate(vecs_dict):
            if i == 0:
                X = vecs_dict[f]
            else:
                X = np.concatenate((X, vecs_dict[f]), axis=0)
            inds += [f] * len(vecs_dict[f])
        T = TSNE(n_components=2).fit_transform(X)

        # scatter TSNE
        classes = set(all_tags)
        for j, tag in enumerate(classes):
            scatter_tsne(T, inds, tag, 'Other', os.path.join(path, 'tsne_results_' + tag), colors[j])
            # scatter_tsne(T, inds, tag, 'Other', os.path.join(path, 'tsne_results_' + tag), color)
    # KDE ----------------------------------------------
    # run KDE to find all pairwise distances between samples
    # run for each of the trained models twice, with and without frequencies

    if RUN_KDE:

        print('KDE')

        for path in [encoder_dir, v_encoder_dir, embedding_dir, v_embedding_dir]:

            if not RUN_AE:
                continue

            if not RUN_EMB:
                continue

            if not RUN_V_AE:
                continue

            if not RUN_V_EMB:
                continue

            if (path == v_encoder_dir or path == v_embedding_dir) and V_S == 'None':
                continue

            # load projections
            vecs_dict = load_representations(path, N_PROJECTIONS)
            if F_S == 'None':
                vecs_dict_fs = 'None'
            else:
                vecs_dict_fs = load_representations_by_freqs(path, N_PROJECTIONS)

            for round in [[vecs_dict, 'no_freqs'], [vecs_dict_fs, 'with_freqs']]:

                if round[0] == 'None':
                    continue

                # caclculate KDE for every pair of samples
                dis_dict = compute_kde_distances(round[0])

                # save distances in csv
                headers = list(dis_dict.keys())
                path_to_csv = path + 'kde_representations_' + round[1] + '.csv'
                kde_to_csv(dis_dict, headers, path_to_csv)
                data_arr, self_arr, headers = csv_to_arr(path_to_csv)

                # plot distances, heatmap without diagonal and bar plot of self-distances
                plot_heatmap(data_arr, self_arr, headers, path, round[1])
                plot_self_bar(self_arr, headers, path, round[1])

    # Projections Properties ----------------------------------------------
    # check repertoire features within the embedded space

    if RUN_PROPS:

        print('projections properties')

        # Public Clones

        path = os.path.join(save_path, parameters["tsne_props_encoder"] + "/")  # projections to use
        public_cdr3_dict = public_cdr3s(root, DEL_S, CDR3_S)  # public cdr3 sequences
        # load projections with their original sequences
        projections_dict = load_representations_and_seqs(path, N_PROJECTIONS)
        projections_arr = []
        for s in projections_dict:
            projections_arr += projections_dict[s]
        center_p = find_center(projections_arr)  # center of all projections in the dataset
        public_distances = distances_from_center(projections_arr, center_p, public_cdr3_dict)  # distance from center

        # plot distances results
        x = 'Publicity'
        y = 'Distance from center'
        distances_df = dict_to_df(public_distances, x, y)
        plot_distances(x, y, distances_df, path + 'public_distance_from_center.png')
        public_densities = self_density(projections_dict, public_cdr3_dict)
        x = 'Publicity'
        y = 'Self-Density'
        densities_df = dict_to_df(public_densities, x, y)
        plot_distances(x, y, densities_df, path + 'public_self_density.png')

        # Sequence Features (AA, Length)

        path = encoder_dir  # projections to use
        # load projections with their original sequences
        projections_dict = load_representations_and_seqs(path, N_PROJECTIONS)
        # split to vectors and sequences
        vecs = []
        seqs = []
        for s in projections_dict:
            vecs += [v[0] for v in projections_dict[s]]
            seqs += [v[1] for v in projections_dict[s]]

        # calculate TSNE
        T = TSNE(n_components=2).fit_transform(np.array(vecs))

        # projections by amino-acids
        n_pos = len(vecs[0])
        cs = ['magenta', 'springgreen', 'cornflowerblue']
        for i, a in enumerate(['Cysteine', 'Proline', 'Glycine']):
            scatter_tsne_aa(T, seqs, a, cs[i], path)
            check_list = []
            for seq in seqs:
                if a[0] in seq[1:]:
                    check_list.append(1)
                else:
                    check_list.append(0)
            ts = []
            ps = []
            for ind in range(n_pos):
                v = []
                x = []
                for j, vec in enumerate(vecs):
                    if check_list[j] == 1:
                        v.append(vec[ind])
                    else:
                        x.append(vec[ind])
                t, p = scipy.stats.ttest_ind(x, v, equal_var=False)
                ts.append(t)
                ps.append(p)
            plot_t_bars(ts, ps, a, cs[i], path)

        # projections by lengths
        scatter_tsne_all_lengths(T, seqs, path)

        # distance from center
        CM = np.average(vecs, axis=0)
        lens = []
        dis = []
        for i in range(len(vecs)):
            lens.append(len(seqs[i]))
            dis.append(np.linalg.norm(vecs[i] - CM))
        df = pd.DataFrame(data={'CDR3 Length': lens, 'Distance From Center': dis})
        scatter_cm(df, path)

    # KL Distances ----------------------------------------------

    if RUN_KL and V_S != 'None':

        print('KL')

        # load all Vs
        vs_samples, all_vs = load_vs_by_samples(root, DEL_S, V_S, N_FOR_DISTANCES)

        # calculate distributions
        dist_dict = dict()
        for file in vs_samples:
            dist_dict[file] = calc_distribution(vs_samples[file], all_vs)

        # compute KL between every two samples
        kl_dict = dict()
        for i, file in enumerate(dist_dict.keys()):
            kl_dict[file] = dict()
            for j, file2 in enumerate(dist_dict.keys()):
                if i == j:
                    kl_dict[file][file2] = np.nan
                elif i < j:
                    kl_dict[file][file2] = KL(dist_dict[file], dist_dict[file2])
                else:
                    kl_dict[file][file2] = kl_dict[file2][file]

        # fill the full matrix of the data and save csv
        with open(os.path.join(save_path, 'kl_distances.csv'), 'w') as csvfile:
            fieldnames = ['file'] + list(kl_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            data_arr = []
            for f in kl_dict:
                data = {'file': f}
                tmp = []
                for f2 in kl_dict:
                    val = kl_dict[f][f2]
                    data[f2] = val
                    tmp.append(val)
                writer.writerow(data)
                data_arr.append(tmp)

        if len(data_arr) > 1:
            # plot distances
            kl_heatmap(data_arr, list(kl_dict.keys()), os.path.join(save_path, 'kl_distances.png'))

    # ED Distances ----------------------------------------------

    if RUN_ED:

        print('ED')

        # load all sequences
        cdr3_samples = load_seqs_by_samples(root, DEL_S, CDR3_S, N_FOR_DISTANCES)

        # average min edit distance (levenshtein distance)
        all_ds, files_list = cxc_similarity(cdr3_samples)

        if len(all_ds) > 1:
            # plot distances
            plot_similarity_mat(all_ds, files_list, save_path)

        # fill the full matrix of the data and save to csv
        with open(os.path.join(save_path, 'ed_distances.csv'), 'w') as csvfile:
            fieldnames = ['file'] + files_list
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, val in enumerate(all_ds):
                file = files_list[i]
                data = {'file': file}
                for j, file2 in enumerate(fieldnames[1:]):
                    data[file2] = val[j]
                writer.writerow(data)

    # MDS ----------------------------------------------
    # run MDS for each KDE matrix

    if RUN_MDS:

        print('MDS')

        for path in [encoder_dir, v_encoder_dir, embedding_dir]:

            if not RUN_AE:
                continue

            if not RUN_EMB:
                continue

            if not RUN_V_AE:
                continue

            if path == v_encoder_dir and V_S == 'None':
                continue

            flag = True
            if F_S == 'None':
                flag = False

            for round in ['no_freqs', 'with_freqs']:

                if round == 'with_freqs' and not flag:
                    continue

                path_to_csv = path + 'kde_representations_' + round + '.csv'
                data_arr, self_arr, headers = csv_to_arr(path_to_csv)
                headers = [tag_per_data_set(file, root) for file in headers]

                # find minimum to remove baseline
                check_min = []
                for i in range(len(data_arr)):
                    for j in range(len(data_arr)):
                        if i == j:
                            data_arr[i][j] = 0.0
                        else:
                            check_min.append(data_arr[i][j])

                baseline = np.array(check_min).min()
                for i in range(len(data_arr)):
                    for j in range(len(data_arr)):
                        if i != j:
                            data_arr[i][j] -= baseline
                data_arr = np.array(data_arr)

                # calculate MDS
                model = MDS(n_components=3, random_state=1)
                out = model.fit_transform(data_arr)

                # scatter MDS
                scatter_3D_MDS(out, headers, path, round)

    # remove_pickle_files(save_path)

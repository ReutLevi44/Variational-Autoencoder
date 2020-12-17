import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d
import seaborn as sns

from utils_preprocessing import tag_per_data_set


# plot loss
def plot_loss(_loss, _val_loss, path):
    plt.figure()
    _epochs = range(len(_val_loss))
    plt.plot(_epochs, _loss, 'bo', label='Training loss', color='mediumaquamarine')
    plt.plot(_epochs, _val_loss, 'b', label='Validation loss', color='cornflowerblue')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path + 'loss_plot')
    plt.savefig('loss_plot')
    plt.clf()
    plt.close()


def scatter_tsne(data, inds, tag, neg_str, path, color1):
    color2 = 'lightsteelblue'
    for i, val in enumerate(data):
        if tag_per_data_set(inds[i], path) == tag:
            c = color1
            m = 'D'
            s = 7
        else:
            c = color2
            m = '.'
            s = 10
        plt.scatter(val[0], val[1], color=c, marker=m, s=s)
    plt.title('Autoencoder Projections TSNE')
    plt.tight_layout()
    patches = [mpatches.Patch(color='red', label=tag),  # color=color1
               mpatches.Patch(color=color2, label=neg_str)]
    plt.legend(handles=patches, fontsize='small', loc=2)
    plt.savefig(path)
    plt.clf()
    plt.close()


def plot_heatmap(data, diagonal, headers, s, round_s):
    sns.set(font_scale=0.5)
    for i, row in enumerate(data):
        for j in range(len(row)):
            if i != j:
                data[i][j] = data[i][j] / diagonal[i]

    sns.heatmap(data, xticklabels=headers, yticklabels=headers, cmap=sns.diverging_palette(10, 220, sep=80, n=7),
                annot_kws={"size": 10})
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title(s + 'KDE Representations Distances')
    plt.tight_layout()
    plt.savefig(s + 'kde_representations_distances_' + round_s + '.png')
    plt.clf()
    plt.close()


def plot_self_bar(y_list, x_ticks, s, round_s):
    pos = np.arange(len(x_ticks))
    plt.bar(pos, [np.log(y) for y in y_list], align='center', alpha=0.5)
    plt.xticks(pos, x_ticks, rotation='vertical')
    plt.ylabel('KDE Distances')
    plt.title(s + 'KDE Within The Diagonal')
    plt.tight_layout()
    plt.savefig(s + 'kde_self_bar_' + round_s + '.png')
    plt.clf()
    plt.close()


def plot_distances(my_x, my_y, my_df, my_path, min=0, max=0):
    sns.boxplot(x=my_x, y=my_y, data=my_df, color="royalblue", boxprops=dict(alpha=.7))
    if min != 0 and max != 0:
        plt.ylim(min, max)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig(my_path)
    plt.clf()
    plt.close()


# scatter projections by amino acid
def scatter_tsne_aa(data, seqs, aa, color, path):
    categories = {aa: 'red', 'Other': 'lightsteelblue'}  # aa: color
    for i, val in enumerate(data):
        if aa[0] in seqs[i][1:]:
            key = aa
            m = 'D'
            s = 7
        else:
            key = 'Other'
            m = '.'
            s = 10
        c = categories[key]
        plt.scatter(val[0], val[1], color=c, marker=m, s=s)
    plt.title('Projections TSNE ' + aa)
    plt.tight_layout()
    patches = []
    for key in categories:
        patches.append(mpatches.Patch(color=categories[key], label=key))
    plt.legend(handles=patches, fontsize='small', loc=2)
    plt.savefig(path + 'projections_tsne_' + aa)
    plt.clf()
    plt.close()


# scatter projections by lengths
def scatter_tsne_all_lengths(data, seqs, path):
    cs = ['dimgray', 'darkgray', 'lightgray',
          'lightcoral', 'coral', 'darkorange', 'orange', 'darkgoldenrod', 'olive', 'yellowgreen', 'lawngreen',
          'lightgreen', 'mediumaquamarine',
          'c', 'cadetblue', 'skyblue', 'darkblue', 'slateblue', 'rebeccapurple', 'darkviolet', 'violet', 'pink',
          'crimson', 'r', 'brown', 'maroon']
    lens = list(set([len(i) for i in seqs]))
    cs_dict = {}
    for i in range(len(lens)):
        cs_dict[lens[i]] = cs[i]
    for i, val in enumerate(data):
        n = len(seqs[i])
        if n not in cs_dict:
            cs_dict[n] = cs.pop()
        c = cs_dict[n]
        plt.scatter(val[0], val[1], c=c, marker='.', s=10)
    plt.title('Projections TSNE , Lengths')
    plt.tight_layout()
    patches = []
    for key in np.sort(list(cs_dict.keys())):
        patches.append(mpatches.Patch(color=cs_dict[key], label=key))
    plt.legend(handles=patches, fontsize='small', loc=2)
    plt.savefig(path + 'projections_tsne_lengths')
    plt.clf()
    plt.close()


def scatter_cm(d, path):
    sns.boxplot(x='CDR3 Length', y='Distance From Center', data=d, color="crimson", boxprops=dict(alpha=.7))
    plt.title('Projections Radius By CDR3 Length')
    plt.tight_layout()
    plt.savefig(path + 'projections_radius_cdr3_length_boxplot')
    plt.clf()
    plt.close()


# t-test br plot per amino acid
def plot_t_bars(ts_, ps_, aa, color, path):
    sns.set(font_scale=0.8)
    pos = np.arange(len(ts_))
    labels = []
    for p in ps_:
        if p < 0.001:
            labels.append('***')
        else:
            if p < 0.01:
                labels.append('**')
            else:
                if p < 0.05:
                    labels.append('*')
                else:
                    labels.append('')
    plt.bar(pos, ts_, align='center', alpha=0.5, color=color)
    gaps = []
    for t in ts_:
        if t < 0:
            gaps.append(-1)
        else:
            gaps.append(0.1)
    for i in range(len(pos)):
        plt.text(i, ts_[i] + gaps[i], s=labels[i], ha='center', fontsize=8)
    plt.xticks(pos, pos, rotation='vertical')
    plt.ylabel('t Value')
    plt.title('Projections ' + aa + ' t-Test')
    plt.tight_layout()
    plt.savefig(path + 'projections_t-test_' + aa)
    plt.clf()
    plt.close()


# plot KL distance heatmap
def kl_heatmap(data, headers, path):
    sns.set(font_scale=0.4)
    ax = sns.heatmap(data, xticklabels=headers, yticklabels=headers, cmap='coolwarm', annot_kws={"size": 3})
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Vs distributions KL test')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()


# plot ED self-distances
def plot_diagonal(x, files, path):
    sns.set(font_scale=0.8)
    pos = np.arange(len(x))
    plt.bar(pos, x, align='center', alpha=0.5, color='b')
    plt.xticks(pos, files, rotation='vertical')
    plt.ylabel('ED Distance')
    plt.title('ED diagonal')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'ed_distances_diagonal.png'))
    plt.clf()
    plt.close()


# heatmap plot of similarity between cdr3 sets of different compartments
def plot_similarity_mat(data_, comps, path):
    # nan at diagonal
    diag = []
    data = []
    for i in range(len(data_)):
        row = []
        for j in range(len(data_)):
            if i == j:
                diag.append(data_[i][j])
                val = np.nan
            else:
                val = data_[i][j]
            row.append(val)
        data.append(row)
    plot_diagonal(diag, comps, path)
    sns.set(font_scale=0.5)
    ax = sns.heatmap(data, xticklabels=comps, yticklabels=comps, cmap='coolwarm', annot_kws={"size": 3})
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Min Average Edit Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'ed_distances.png'))
    plt.clf()
    plt.close()


def scatter_3D_MDS(results, samples_list, my_path, s_round):
    colors = ['gold', 'm', 'blue', 'lime', 'orchid', 'grey', 'r']
    tags_set = set(samples_list)
    colors_dict = {t: colors[i] for i, t in enumerate(tags_set)}
    colors = [colors_dict[s] for s in samples_list]
    ax = plt.axes(projection='3d')
    for x, y, z, c in zip(results[:, 0], results[:, 1], results[:, 2], colors):
        ax.scatter3D([x], [y], [z], c=c, s=15, marker='x')
    patches = []
    for key in colors_dict:
        patches.append(mpatches.Patch(color=colors_dict[key], label=key))
    plt.legend(handles=patches, fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('KDE MDS')
    plt.xlabel('First MDS', fontsize=10)
    plt.ylabel('Second MDS', fontsize=10)
    plt.tight_layout()
    plt.savefig(my_path + s_round + '_MDS.png')
    plt.clf()
    plt.close()

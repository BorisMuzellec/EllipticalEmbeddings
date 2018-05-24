from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import pandas as pd
import pickle as pkl
import cupy as cp
import os

from scipy.stats import rankdata
from cupy.cuda import Device

import copy
import argparse
import tqdm

import itertools

import utils as wb


parser = argparse.ArgumentParser(description='Evaluate ranks and MAP on hypernym reconstruction')

parser.add_argument('--data', '-d', type=str, help='csv file containing the relations')
parser.add_argument('--embeddings', '-e', type=str,
                    help='folder containing the embeddings')
parser.add_argument('--max_n', type=int, default=None, help = 'number of items to evaluate')
parser.add_argument('--metric', type=str, default="bures_product")
parser.add_argument('--device', type=int, default=7)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()



def get_positive_relation_ranks_and_avg_prec(all_distances, positive_relations):
    """
    Given a numpy array of all distances from an item and indices of its positive relations,
    compute ranks and Average Precision of positive relations.
    Parameters
    ----------
    all_distances : numpy.array (float)
        Array of all distances (floats) for a specific item.
    positive_relations : list
        List of indices of positive relations for the item.
    Returns
    -------
    tuple (list, float)
        The list contains ranks (int) of positive relations in the same order as `positive_relations`.
        The float is the Average Precision of the ranking.
        e.g. ([1, 2, 3, 20], 0.610).
    """
    positive_relation_distances = all_distances[positive_relations]
    negative_relation_distances = np.ma.array(all_distances, mask=False)
    negative_relation_distances.mask[positive_relations] = True
    # Compute how many negative relation distances are less than each positive relation distance, plus 1 for rank
    ranks = (negative_relation_distances < positive_relation_distances[:, np.newaxis]).sum(axis=1) + 1
    map_ranks = np.sort(ranks) + np.arange(len(ranks))
    avg_precision = ((np.arange(1, len(map_ranks) + 1) / np.sort(map_ranks).astype(float)).mean())
    return list(ranks), avg_precision


def evaluate_mean_rank_and_map(hypernym_df, max_n=None):
    """Evaluate mean rank and MAP for reconstruction.
    Parameters
    ----------
    max_n : int or None
        Maximum number of positive relations to evaluate, all if `max_n` is None.
    Returns
    -------
    tuple (float, float)
        Contains (mean_rank, MAP).
        e.g (50.3, 0.31)
    """
    pass


def main(embedding_file=args.embeddings, hypernym_file=args.data, device=args.device, max_n = args.max_n):

    Device(device).use()

    print("Reading embedding file:", embedding_file)

    with open(embedding_file, 'rb') as embed_file:
        embeddings = pkl.load(embed_file)

    means = embeddings['means']
    vars = embeddings["vars"]

    words_to_idx = embeddings['word_to_idx']

    hypernym_df = pd.read_csv(hypernym_file, header=None, sep='\t')
    hypernym_df[2] = hypernym_df[0].apply(lambda x: words_to_idx[x])
    hypernym_df[3] = hypernym_df[1].apply(lambda x: words_to_idx[x])
    hypernym_couples = []

    print("Ranking distances")

    idxs1 = hypernym_df[2].values
    idxs2 = hypernym_df[3].values

    n = np.max(idxs1) + 1
    m = np.max(idxs2) + 1

    idxs = np.arange(m)

    ranks = []
    avg_precision_scores = []

    idxs1 = list(set(idxs1))
    np.random.shuffle(idxs1)

    for i, idx in tqdm.tqdm(enumerate(idxs1)):
        x = [idx] * m
        if args.metric == "distance":
            dists_ = wb.batch_W2(cp.array(means[x]), cp.array(means[idxs]), cp.array(vars[x]), cp.array(vars[idxs]))[0]
            item_distances = cp.asnumpy(dists_)
        elif args.metric == "bures_cosine":
            scores_ = wb.bures_cosine(cp.array(means[x]), cp.array(means[idxs]), cp.array(vars[x]), cp.array(vars[idxs]))
            item_distances = -cp.asnumpy(scores_)
        elif args.metric == "bures_product":
            scores_ = wb.batch_W2(cp.array(means[x]), cp.array(means[idxs]), cp.array(vars[x]), cp.array(vars[idxs]), prod=True)[0]
            item_distances = -cp.asnumpy(scores_)

        assert(not (np.isnan(item_distances.any()) or np.isinf(item_distances).any()))

        item_relations = list(set(hypernym_df[hypernym_df[2] == idx][3]).union(set(hypernym_df[hypernym_df[3] == idx][2])))
        positive_relation_ranks, avg_precision = \
            get_positive_relation_ranks_and_avg_prec(item_distances, item_relations)
        ranks += positive_relation_ranks
        avg_precision_scores.append(avg_precision)
        if max_n is not None and i > max_n:
            break
        if args.verbose:
            print('Positive relation distances: ', item_distances[item_relations])
            print('Ranks: ', positive_relation_ranks)
            print('AP:', avg_precision)

        if (i+1) % 10000 == 0:
            print("Rank moving average: %f" % (np.mean(ranks)))
            print("MAP moving average: %f" % (np.mean(avg_precision_scores)))

    print('Mean rank: %f\t MAP: %f\t Rank std: %f\t AP std: %f' % (np.mean(ranks), np.mean(avg_precision_scores), np.std(ranks), np.std(avg_precision_scores)))

    with open(os.path.join(os.path.dirname(embedding_file), "rank_summary"), "a") as result_file:
        result_file.write("File: %s \t Sample size %u\t Metric: %s\t Full Mean rank: %f\t MAP: %f\t Rank std: %f\t AP std: %f\n" % (
        embedding_file.split('/')[-1], len(avg_precision_scores), args.metric, np.mean(ranks), np.mean(avg_precision_scores), np.std(ranks), np.std(avg_precision_scores)))

    if args.max_n is None:
        with open(embedding_file + "_ranks.pkl", "wb") as ranks_file:
            pkl.dump(ranks, ranks_file)
        with open(embedding_file + "_precisons.pkl", "wb") as precision_file:
            pkl.dump(avg_precision_scores, precision_file)

if __name__ == '__main__':
    main()

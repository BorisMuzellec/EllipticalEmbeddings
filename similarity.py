import numpy as np
import cupy as cp
import pandas as pd
import scipy
import os
import pickle as pkl
import re

import utils as wb
from embeddings import EllEmbeddings

from sklearn.metrics import f1_score

'''
The code for similarity evaluation was in large part obtained from this repository: https://github.com/benathi/word2gm
'''


## 1. Evaluation Data Loading
def load_SimLex999(filepath='/home/boris/word2gm/evaluation_data/SimLex-999/SimLex-999.txt'):
    _fpath = filepath if filepath is not None else os.environ['SIMLEX999_FILE']
    df = pd.read_csv(_fpath, delimiter='\t')
    word1 = df['word1'].tolist()
    word2 = df['word2'].tolist()
    score = df['SimLex999'].tolist()
    assert len(word1) == len(word2) and len(word1) == len(score)
    return word1, word2, score


def load_data_format1(filename='EN-MC-30.txt', delim='\t', verbose=False):
    if verbose: print('Loading file', filename)
    fpath = os.path.join('/home/boris/word2gm/evaluation_data/multiple_datasets', filename)
    df = pd.read_csv(fpath, delimiter=delim, header=None)
    word1 = df[0].tolist()
    word2 = df[1].tolist()
    score = df[2].tolist()
    assert len(word1) == len(word2) and len(word1) == len(score)
    return word1, word2, score


def load_MC():
    return load_data_format1(filename='EN-MC-30.txt')


def load_MEN():
    return load_data_format1(filename='EN-MEN-TR-3k.txt', delim=' ')


def load_Mturk287():
    return load_data_format1(filename='EN-MTurk-287.txt')


def load_Mturk771():
    return load_data_format1(filename='EN-MTurk-771.txt', delim=' ')


def load_RG():
    return load_data_format1(filename='EN-RG-65.txt')


def load_RW_Stanford():
    return load_data_format1(filename='EN-RW-STANFORD.txt')


def load_WS_all():
    return load_data_format1(filename='EN-WS-353-ALL.txt')


def load_WS_rel():
    return load_data_format1(filename='EN-WS-353-REL.txt')


def load_WS_sim():
    return load_data_format1(filename='EN-WS-353-SIM.txt')


def load_YP():
    return load_data_format1(filename='EN-YP-130.txt', delim=' ')


def calculate_correlation(data_loader, w2g, verbose=True, lower=False, metric='bures_cosine', embedds = 'input', numIters=50):
    #### data_loader is a function that returns 2 lists of words and the scores
    #### metric is a function that takes w1, w2 and calculate the score

    word1, word2, targets = data_loader()

    if lower:
        word1 = [str.encode(word.lower()) for word in word1]
        word2 = [str.encode(word.lower()) for word in word2]

    mask = [(w1 in w2g.words_to_idxs and w2 in w2g.words_to_idxs) for (w1,w2) in zip(word1, word2)]
    word1, word2, targets = np.array(word1)[mask], np.array(word2)[mask], np.array(targets)[mask]

    distinct_words = set(np.concatenate([word1,word2]))
    ndistinct = len(distinct_words)
    nwords_dict = len([w in w2g.words_to_idxs for w in distinct_words])
    if lower:
        nwords_dict = len([w.lower() in w2g.words_to_idxs for w in distinct_words])
    if verbose: print('# of pairs {} # words total {} # words in dictionary {}({}%)' \
        .format(len(word1), ndistinct, nwords_dict, 100 * nwords_dict / (1. * ndistinct)))

    word1_idxs = [w2g.words_to_idxs[w] for w in word1]
    word2_idxs = [w2g.words_to_idxs[w] for w in word2]

    scores = np.zeros((len(word1_idxs)))

    if embedds == 'input':
        if metric == 'bures_distance':
            scores = -wb.batch_W2(w2g.means[word1_idxs], w2g.means[word2_idxs], w2g.vars[word1_idxs], w2g.vars[word2_idxs], numIters=numIters)[0]
        elif metric == 'bures_cosine':
            scores = wb.bures_cosine(w2g.means[word1_idxs], w2g.means[word2_idxs], w2g.vars[word1_idxs], w2g.vars[word2_idxs], numIters=numIters)
        elif metric == 'bures_product':
            scores = wb.batch_W2(w2g.means[word1_idxs], w2g.means[word2_idxs], w2g.vars[word1_idxs], w2g.vars[word2_idxs], prod=True, numIters=numIters)[0]
        elif metric == 'kl':
            scores = -wb.diag_kl(w2g.means[word1_idxs], w2g.means[word2_idxs], w2g.vars[word1_idxs],
                                 w2g.vars[word2_idxs])
    else:

        if metric == 'bures_distance':
            scores= -wb.batch_W2(w2g.c_means[word1_idxs], w2g.c_means[word2_idxs], w2g.c_vars[word1_idxs], w2g.c_vars[word2_idxs], numIters=numIters)[0]
        elif metric == 'bures_cosine':
            scores = wb.bures_cosine(w2g.c_means[word1_idxs], w2g.c_means[word2_idxs], w2g.c_vars[word1_idxs],
                                     w2g.c_vars[word2_idxs], numIters=numIters)
        elif metric == 'bures_product':
            scores = \
            wb.batch_W2(w2g.c_means[word1_idxs], w2g.c_means[word2_idxs], w2g.c_vars[word1_idxs], w2g.c_vars[word2_idxs],
                        prod=True, numIters=numIters)[0]
        elif metric == 'kl':
            scores = -wb.batch_kl(w2g.c_means[word1_idxs], w2g.c_means[word2_idxs], w2g.c_vars[word1_idxs],
                                  w2g.c_vars[word2_idxs])

    scores = cp.asnumpy(scores)
    spr = scipy.stats.spearmanr(scores, targets)
    if verbose: print('Spearman correlation is {} with pvalue {}'.format(spr.correlation, spr.pvalue))
    pear = scipy.stats.pearsonr(scores, targets)
    if verbose: print('Pearson correlation', pear)
    spr_correlation = spr.correlation
    pear_correlation = pear[0]
    if np.any(np.isnan(scores)):
        spr_correlation = np.NAN
        pear_correlation = np.NAN
    return scores, spr_correlation, pear_correlation


eval_datasets = [load_SimLex999, load_WS_all, load_WS_sim, load_WS_rel,
                 load_MEN, load_MC, load_RG, load_YP,
                 load_Mturk287, load_Mturk771,
                 load_RW_Stanford]

eval_datasets_names_full = []
for dgen in eval_datasets:
    eval_datasets_names_full.append(dgen.__name__[5:])
eval_datasets_names = ['SL', 'WS', 'WS-S', 'WS-R', 'MEN',
                       'MC', 'RG', 'YP', 'MT-287', 'MT-771', 'RW']


# performs quantitative evaluation in a batch
def quantitative_eval(model_names, vocab_path, prefix_dir='', metrics = ['bures_cosine'],
                      lower=False, verbose=False, type = 'full', embedds = 'input', device=0, numIters=20, output_path=None):
    # model_names is a list of pairs (model_abbreviation, save_path)

    spearman_corrs = pd.DataFrame()
    spearman_corrs['Dataset'] = eval_datasets_names
    # folder path of this code
    # allow it to be called from other directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # for i, (model_abbrev, save_path) in enumerate(model_names):
    for i, save_path in enumerate(model_names):
        if verbose: print('Processing', save_path)
        if True:
            if verbose:print('dir path =', dir_path)
            save_path_full = os.path.join(dir_path, prefix_dir, save_path)
            w2g = EllEmbeddings(vocab_path, save_path, type=type, device=device)
            for metric in metrics:
                results = []
                for dgen in eval_datasets:
                    if verbose: print('data', dgen.__name__)
                    _, sp, pe = calculate_correlation(dgen,  w2g, metric=metric, lower=lower, verbose=verbose, embedds = embedds, numIters=numIters)
                    results.append(sp * 100)
                colname = '{}/{}'.format(save_path, metric)
                spearman_corrs[colname] = results
    if output_path is not None:
        with open(output_path, 'wb') as output_file:
            pkl.dump(spearman_corrs, output_file)
    return spearman_corrs

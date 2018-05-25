
import numpy as np
import cupy as cp

import utils as wr

import os
import pickle as pkl


FORMAT = '%(asctime)-15s %(message)s'

class EllEmbeddings():
    def __init__(self, vocab_dir, embeddings_file, device=0):

        cp.cuda.Device.use(cp.cuda.Device(device))

        self.vocab_dir = vocab_dir
        self.embeddings_file = embeddings_file

        self.vocab_words= self.load_idx_to_word(self.vocab_dir)
        self.words_to_idxs = self.load_word_to_idxs(self.vocab_dir)

        embeddings = self.load_embeddings(self.embeddings_file)

        self.means = cp.array(embeddings['means'])
        self.c_means = cp.array(embeddings['c_means'])

        self.vars = cp.array(embeddings['vars'])
        self.c_vars = cp.array(embeddings['c_vars'])

        self.n_points, self.n_dim = self.means.shape[0], self.means.shape[1]

    def load_idx_to_word(self, vocab_dir):

        with open(os.path.join(vocab_dir, 'vocab_words.pkl'), 'rb') as vocabfile:
            vocab_words = pkl.load(vocabfile)

        return vocab_words

    def load_word_to_idxs(self, vocab_dir):

        with open(os.path.join(vocab_dir, 'words_to_idxs.pkl'), 'rb') as vocabfile:
            word_dict = pkl.load(vocabfile)

        return word_dict

    def load_embeddings(self, embeddings_file):

        with open(embeddings_file, 'rb') as embeddfile:
            embeddings = pkl.load(embeddfile)

        return embeddings

    def nearest_neighbours(self, word, k=10, metric='bures_distance'):
        n = len(self.means)
        widx = [self.words_to_idxs[word]] * n

        if metric == 'bures_distance':
            dists = wr.batch_W2(cp.array(self.c_means)[widx], cp.array(self.c_means), cp.array(self.c_vars)[widx], cp.array(self.c_vars), numIters=20)[0]
        elif metric == 'bures_product':
            dists = wr.batch_W2(cp.array(self.c_means)[widx], cp.array(self.c_means), cp.array(self.c_vars)[widx],
                                cp.array(self.c_vars), numIters=20, prod=True)[0]
            dists = -dists
        elif metric == 'bures_cosine':
            dists = wr.bures_cosine(cp.array(self.c_means)[widx], cp.array(self.c_means), cp.array(self.c_vars)[widx],
                                    cp.array(self.c_vars), numIters=20)
            dists = -dists

        idxs = np.argsort(cp.asnumpy(dists))[:k]

        for i in range(k):
            print(self.vocab_words[idxs[i]])


    def compose(self, from_word, to_word, test_word, k=10, metric='bures_distance'):
        n = len(self.means)
        t, T = self.get_push_forward(from_word, to_word)

        test_sigma = cp.array(self.c_vars[self.words_to_idxs[test_word]])

        mu = cp.array(self.c_means[self.words_to_idxs[test_word]]) + t
        sigma = cp.matmul(T, cp.matmul(test_sigma.reshape(-1, test_sigma.shape[0], test_sigma.shape[1]), T))

        if metric == 'bures_distance':
            dists = wr.batch_W2(mu.reshape(1, -1).repeat(n, axis=0), cp.array(self.c_means), sigma.repeat(n, axis=0), cp.array(self.c_vars), numIters=20)[0]
        elif metric == 'product':
            dists = wr.batch_W2(mu.reshape(1, -1).repeat(n, axis=0), cp.array(self.c_means), sigma.repeat(n, axis=0), cp.array(self.c_vars), numIters=20, prod=True)[0]
            dists = -dists
        elif metric == 'bures_cosine':
            dists = wr.bures_cosine(mu.reshape(1, -1).repeat(n, axis=0), cp.array(self.c_means), sigma.repeat(n, axis=0), cp.array(self.c_vars), numIters=20)
            dists = -dists

        idxs = np.argsort(cp.asnumpy(dists))[:k]

        for i in range(k):
            print(self.vocab_words[idxs[i]])

    def get_push_forward(self, from_word, to_word):
        from_mu = self.c_means[self.words_to_idxs[from_word]]
        from_sigma = self.c_vars[self.words_to_idxs[from_word]]
        to_mu = self.c_means[self.words_to_idxs[to_word]]
        to_sigma = self.c_vars[self.words_to_idxs[to_word]]

        return cp.array(to_mu - from_mu), wr.batch_Tuv2(cp.array(from_sigma).reshape(1, from_sigma.shape[0], from_sigma.shape[1]), cp.array(to_sigma).reshape(1, to_sigma.shape[0], to_sigma.shape[1]), numIters=20)

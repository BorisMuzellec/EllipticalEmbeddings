import logging

import cupy as cp
import utils as wb
from cupy.cuda import Device


import pickle as pkl

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


class EllSoftmax():
    def __init__(self, n_points, n_dim, lr=1E-1, num_neg=1, window_size=10,
                 num_sqrt_iters=5, Cn=1, lbda = 1E-8,
                 scale=0.5, embedding_file=None, optim='rmsprop', epsilon=1E-8, unknown_words=True,
                 sep_input_output=True):
        self.n_points = n_points
        self.n_dim = n_dim
        self.num_neg = num_neg
        self.window_size = window_size
        self.loss = 0
        self.Cn = Cn
        self.lr = lr
        self.lbda = lbda
        self.num_sqrt_iters = num_sqrt_iters
        self.scale = cp.sqrt(3.0 / self.n_dim) * scale
        self.optim = optim
        self.unknown_words = unknown_words
        self.epsilon = epsilon

        if embedding_file is None:
            # Initialize the embeddings
            self.means = 2 * self.scale * (cp.random.rand(n_points, n_dim) - 0.5)
            self.vars, _ = wb.batch_sqrtm(wb.wishart(n_points, n_dim, 2 * n_dim) / (2 * n_dim))
            self.c_means = 2 * self.scale * (cp.random.rand(n_points, n_dim) - 0.5)
            self.c_vars, _ = wb.batch_sqrtm(wb.wishart(n_points, n_dim, 2 * n_dim) / (2 * n_dim))


        else:
            with open(embedding_file) as embed_file:
                embedds = pkl.load(embed_file)
                self.means = cp.array(embedds['means'])
                self.c_means = cp.array(embedds['c_means'])
                self.vars = cp.array(embedds['vars'])
                self.c_vars = cp.array(embedds['c_vars'])

        if self.unknown_words:
            self.means[0] = cp.zeros(n_dim)
            self.c_means[0] = cp.zeros(n_dim)
            self.vars[0] = cp.zeros((n_dim, n_dim))
            self.c_vars[0] = cp.zeros((n_dim, n_dim))


        # The unknown word should have a 0 embedding
        self.means_adagrad = cp.zeros_like(self.means)
        self.c_means_adagrad = cp.zeros_like(self.c_means)
        self.vars_adagrad = cp.zeros_like(self.vars)
        self.c_vars_adagrad = cp.zeros_like(self.c_vars)

        if not sep_input_output:
            self.c_means = self.means
            self.c_vars = self.vars
            self.c_means_adagrad = self.means_adagrad
            self.c_vars_adagrad = self.vars_adagrad


    def compute_loss(self, i, j, neg_i, neg_j):

        xi = self.means[i]
        lvi = self.vars[i]

        xj = self.c_means[j]
        lvj = self.c_vars[j]

        self.vi = wb.to_full(lvi) + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(i), axis=0)
        self.vj = wb.to_full(lvj) + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(j), axis=0)

        neg_xi = self.means[neg_i]
        lneg_vi = self.vars[neg_i]

        neg_xj = self.c_means[neg_j]
        lneg_vj = self.c_vars[neg_j]

        self.neg_vi = wb.to_full(lneg_vi) + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(neg_i), axis=0)
        self.neg_vj = wb.to_full(lneg_vj) + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(neg_j), axis=0)

        neg_wij, self.inv_n_ij, self.v_n_i_s, self.inv_v_n_i_s, self.mid_n_ij = wb.batch_W2(neg_xi, neg_xj, self.neg_vi,
                                                                                            self.neg_vj,
                                                                                            Cn=self.Cn,
                                                                                            numIters=self.num_sqrt_iters,
                                                                                            prod=True)

        wij, self.inv_ij, self.v_i_s, self.inv_v_i_s, self.mid_ij = wb.batch_W2(xi, xj, self.vi, self.vj,
                                                                                Cn=self.Cn,
                                                                                numIters=self.num_sqrt_iters,
                                                                                sU=self.v_n_i_s[::self.num_neg],
                                                                                inv_sU=self.inv_v_n_i_s[::self.num_neg],
                                                                                prod=True)

        n_maxs = neg_wij.reshape(-1, self.num_neg).max(axis=1)
        maxs = cp.maximum(n_maxs, wij)
        self.exp_ij = cp.exp(wij - maxs)
        self.neg_exp = cp.exp(neg_wij - maxs.repeat(self.num_neg))

        softmax = maxs + cp.log(self.exp_ij + self.neg_exp.reshape(-1, self.num_neg).sum(axis=1))

        losses = softmax - wij
        loss_wnet = losses.sum()

        self.norm_factor = self.neg_exp.reshape(-1, self.num_neg).sum(axis=1) + self.exp_ij
        self.loss = loss_wnet.sum()

        return self.loss

    def m_grad(self, i, j, neg_i, neg_j, exp_ij, norm_factor, neg_exp):
        xi, xj = self.means[i], self.c_means[j]
        neg_xi, neg_xj = self.means[neg_i], self.c_means[neg_j]
        pos = xj
        neg = neg_xj * neg_exp.reshape(-1, 1)

        # grad_i, grad_j, grad_nj
        return (- pos + (
                pos * exp_ij.reshape(-1, 1) + neg.reshape(-1, self.num_neg, self.n_dim).sum(axis=1)) /
                 norm_factor.reshape(-1, 1)).reshape(-1, self.window_size, self.n_dim).sum(axis=1), \
               ( - xi + xi * exp_ij.reshape(-1, 1) / norm_factor.reshape(-1, 1)), \
               ( + neg_xi * neg_exp.reshape(-1, 1) / norm_factor.repeat(
                   self.num_neg).reshape(-1, 1))

    #TODO: rename variables neg_i, neg_j, etc.

    def v_grad(self, i, j, neg_i, neg_j, neg_vi, neg_vj, vi, vj, v_i_s, v_n_i_s, inv_v_i_s, inv_v_n_i_s, mid_ij, mid_n_ij, inv_ij, inv_n_ij,
               exp_ij, norm_factor, neg_exp):
        lvi, lvj = self.vars[i], self.c_vars[j]
        lneg_vi, lneg_vj = self.vars[neg_i], self.c_vars[neg_j]
        pos_i_ = wb.batch_log2(vi, vj, mid=mid_ij, inv_sU=inv_v_i_s, numIters=self.num_sqrt_iters, prod = True)
        neg_i_ = wb.batch_log2(neg_vi, neg_vj, mid=mid_n_ij, inv_sU=inv_v_n_i_s, numIters=self.num_sqrt_iters, prod=True) * neg_exp.reshape(-1, 1, 1)
        pos_j_ = wb.batch_log(vj, vi, sV=v_i_s, inv=inv_ij, numIters = self.num_sqrt_iters, prod = True)
        neg_j_ = wb.batch_log(neg_vj, neg_vi, sV=v_n_i_s, inv=inv_n_ij, numIters=self.num_sqrt_iters, prod = True)


        pos_i = cp.matmul(pos_i_, lvi)
        pos_j = cp.matmul(pos_j_, lvj)


        return ( - pos_i + (
                         pos_i * exp_ij.reshape(-1, 1, 1) + (cp.matmul(neg_i_, lneg_vi)).reshape(-1, self.num_neg, self.n_dim, self.n_dim).sum(
                     axis=1)) /
                 norm_factor.reshape(-1, 1, 1)).reshape(-1, self.window_size, self.n_dim, self.n_dim).sum(
                   axis=1), \
               (- pos_j + pos_j * exp_ij.reshape(-1, 1, 1) / norm_factor.reshape(-1, 1, 1)), \
               (cp.matmul(neg_j_, lneg_vj)) * neg_exp.reshape(-1, 1, 1) / self.norm_factor.repeat(self.num_neg).reshape(-1, 1, 1)


    def SGD_update(self, i, j, neg_i, neg_j):

        # Means gradients
        m_grad_i, m_grad_j, m_n_grad_j = self.m_grad(i, j, neg_i, neg_j, exp_ij=self.exp_ij, norm_factor=self.norm_factor,
                                                     neg_exp=self.neg_exp)

        # Variances gradient
        v_grad_i, v_grad_j, v_n_grad_j = self.v_grad(i, j, neg_i, neg_j, neg_vi=self.neg_vi, neg_vj=self.neg_vj,
                                                     vi=self.vi, vj=self.vj,exp_ij=self.exp_ij, norm_factor=self.norm_factor,
                                                     neg_exp=self.neg_exp, v_i_s = self.v_i_s,
                                                     v_n_i_s = self.v_n_i_s, inv_v_i_s = self.inv_v_i_s,
                                                    inv_v_n_i_s = self.inv_v_n_i_s, mid_ij = self.mid_ij, mid_n_ij = self.mid_n_ij,
                                                    inv_ij = self.inv_ij, inv_n_ij = self.inv_n_ij)

        m_grad_i_acc, v_grad_i_acc, i_idxs = wb.sum_by_group(m_grad_i, v_grad_i, i)
        m_grad_j_acc, v_grad_j_acc, j_idxs = wb.sum_by_group(cp.concatenate([m_grad_j, m_n_grad_j]), cp.concatenate([v_grad_j, v_n_grad_j]), cp.concatenate([j, neg_j]))


        m_grad_i_acc = cp.array(m_grad_i_acc)
        m_grad_j_acc = cp.array(m_grad_j_acc)
        v_grad_i_acc = cp.array(v_grad_i_acc)
        v_grad_j_acc = cp.array(v_grad_j_acc)

        if self.optim == 'adagrad':

            self.means_adagrad[i_idxs] += m_grad_i_acc**2
            self.c_means_adagrad[j_idxs] += m_grad_j_acc**2
            self.c_means_adagrad[neg_j] += m_n_grad_j**2
            self.vars_adagrad[i_idxs] += v_grad_i_acc**2
            self.c_vars_adagrad[j_idxs] += v_grad_j_acc**2
            self.c_vars_adagrad[neg_j] += v_n_grad_j**2

            # Means updates
            self.means[i_idxs] -= self.lr * m_grad_i_acc / cp.sqrt(self.means_adagrad[i_idxs] + self.epsilon)
            self.c_means[j_idxs] -= self.lr * m_grad_j_acc / cp.sqrt(self.c_means_adagrad[j_idxs] + self.epsilon)
            self.c_means[neg_j] -= self.lr * m_n_grad_j / cp.sqrt(self.c_means_adagrad[neg_j] + self.epsilon)

            self.vars[i_idxs] -= self.lr * self.Cn * v_grad_i_acc / cp.sqrt(self.vars_adagrad[i_idxs] + self.epsilon)
            self.c_vars[j_idxs] -= self.lr * self.Cn * v_grad_j_acc / cp.sqrt(self.c_vars_adagrad[j_idxs] + self.epsilon)
            self.c_vars[neg_j] -= self.lr * self.Cn * v_n_grad_j / cp.sqrt(self.c_vars_adagrad[neg_j] + self.epsilon)

        elif self.optim == 'rmsprop':

            self.means_adagrad[i_idxs] = 0.9 * self.means_adagrad[i_idxs] + 0.1 * m_grad_i_acc**2
            self.c_means_adagrad[j_idxs] = 0.9 * self.c_means_adagrad[j_idxs] + 0.1 * m_grad_j_acc**2
            self.c_means_adagrad[neg_j] = 0.9 * self.c_means_adagrad[neg_j] + 0.1 * m_n_grad_j**2
            self.vars_adagrad[i_idxs] = 0.9 * self.vars_adagrad[i_idxs] + 0.1 * v_grad_i_acc**2
            self.c_vars_adagrad[j_idxs] = 0.9 * self.c_vars_adagrad[j_idxs] + 0.1 * v_grad_j_acc**2
            self.c_vars_adagrad[neg_j] = 0.9 * self.c_vars_adagrad[neg_j] + 0.1 * v_n_grad_j**2

            # Means updates
            self.means[i_idxs] -= self.lr * m_grad_i_acc / cp.sqrt(self.means_adagrad[i_idxs] + self.epsilon)
            self.c_means[j_idxs] -= self.lr * m_grad_j_acc / cp.sqrt(self.c_means_adagrad[j_idxs] + self.epsilon)
            self.c_means[neg_j] -= self.lr * m_n_grad_j / cp.sqrt(self.c_means_adagrad[neg_j] + self.epsilon)

            self.vars[i_idxs] -= self.lr * self.Cn * v_grad_i_acc / cp.sqrt(self.vars_adagrad[i_idxs] + self.epsilon)
            self.c_vars[j_idxs] -= self.lr * self.Cn * v_grad_j_acc / cp.sqrt(self.c_vars_adagrad[j_idxs] + self.epsilon)
            self.c_vars[neg_j] -= self.lr * self.Cn * v_n_grad_j / cp.sqrt(self.c_vars_adagrad[neg_j] + self.epsilon)

        elif self.optim == 'sgd':

            # Means updates
            self.means[i_idxs] -= self.lr * m_grad_i_acc
            self.c_means[j_idxs] -= self.lr * m_grad_j_acc
            self.c_means[neg_j] -= self.lr * m_n_grad_j

            self.vars[i_idxs] -= self.lr * self.Cn * v_grad_i_acc
            self.c_vars[j_idxs] -= self.lr * self.Cn * v_grad_j_acc
            self.c_vars[neg_j] -= self.lr * self.Cn * v_n_grad_j

        if self.unknown_words:
            self.means[0] = cp.zeros(self.n_dim)
            self.c_means[0] = cp.zeros(self.n_dim)
            self.vars[0] = cp.zeros((self.n_dim, self.n_dim))
            self.c_vars[0] = cp.zeros((self.n_dim, self.n_dim))

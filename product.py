import logging

import cupy as cp
import utils as wb
from cupy.cuda import Device


import pickle as pkl

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


class EllProduct():
    def __init__(self, n_points, n_dim, lr=1E-1, num_neg=1, margin = 0.1, window_size=10,  num_sqrt_iters=5, Cn=1,
                 embedding_file=None, optim = 'adagrad', epsilon = 1E-8, unknown_words=True, sep_input_output=True, var_scale=1.0, lbda = 1E-8):
        self.n_points = n_points
        self.n_dim = n_dim
        self.num_neg = num_neg
        self.window_size = window_size
        self.loss = 0
        self.Cn = Cn
        self.lbda = lbda
        self.margin = margin
        self.lr = lr
        self.num_sqrt_iters = num_sqrt_iters
        self.scale = cp.sqrt(3.0 / self.n_dim)
        self.optim = optim
        self.unknown_words = unknown_words
        self.epsilon = epsilon


        if embedding_file is None:
            # Initialize the embeddings
            self.means = 2 * self.scale * (cp.random.rand(n_points, n_dim) - 0.5)
            self.c_means = 2 * self.scale * (cp.random.rand(n_points, n_dim) - 0.5)

            self.vars, _ = wb.batch_sqrtm(wb.wishart(n_points, n_dim, 2 * n_dim) / (2 * n_dim) * var_scale)
            self.c_vars, _ = wb.batch_sqrtm(wb.wishart(n_points, n_dim, 2 * n_dim) / (2 * n_dim) * var_scale)
            #The unknown word should have a 0 embedding

        else:
            with open(embedding_file, 'rb') as embed_file:
                self.vars = cp.zeros((n_points, n_dim, n_dim))
                self.c_vars = cp.zeros((n_points, n_dim, n_dim))
                embedds = pkl.load(embed_file)
                self.means = cp.array(embedds['means'])
                self.c_means = cp.array(embedds['c_means'])
                self.vars[1:], _ = wb.batch_sqrtm(cp.array(embedds['vars'][1:]) - self.lbda * cp.eye(self.n_dim))
                self.c_vars[1:], _ = wb.batch_sqrtm(cp.array(embedds['c_vars'][1:]) - self.lbda * cp.eye(self.n_dim))


        if self.unknown_words:
            self.means[0] = cp.zeros(n_dim)
            self.c_means[0] = cp.zeros(n_dim)
            self.vars[0] = cp.zeros((n_dim, n_dim))
            self.c_vars[0] = cp.zeros((n_dim, n_dim))

        if self.optim != 'sgd':
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
        # lvi = self.vars[i] + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(i), axis=0)
        lvi = self.vars[i]

        xj = self.c_means[j]
        # lvj = self.c_vars[j] + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(j), axis=0)
        lvj = self.c_vars[j]

        self.vi = wb.to_full(lvi) + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(i), axis=0)
        self.vj = wb.to_full(lvj) + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(j), axis=0) * cp.not_equal(j, 0).reshape(-1, 1, 1)

        # Same thing, with the negative batch
        neg_xi = self.means[neg_i]
        # lneg_vi = self.vars[neg_i] + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(neg_i), axis=0)
        lneg_vi = self.vars[neg_i]

        neg_xj = self.c_means[neg_j]
        # lneg_vj = self.c_vars[neg_j] + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(neg_j), axis=0)
        lneg_vj = self.c_vars[neg_j]

        self.neg_vi = wb.to_full(lneg_vi) + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(neg_i), axis=0)
        self.neg_vj = wb.to_full(lneg_vj) + self.lbda * cp.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim).repeat(len(neg_j), axis=0)


        neg_wij, self.inv_n_ij, self.v_n_i_s, self.inv_v_n_i_s, self.mid_n_ij = wb.batch_W2(neg_xi, neg_xj, self.neg_vi, self.neg_vj,
                                                                                            Cn=self.Cn, numIters = self.num_sqrt_iters, prod=True)



        wij, self.inv_ij, self.v_i_s, self.inv_v_i_s, self.mid_ij= wb.batch_W2(xi, xj, self.vi, self.vj,
                                                                               Cn=self.Cn, numIters=self.num_sqrt_iters,
                                                                               sU=self.v_n_i_s[::self.num_neg],
                                                                               inv_sU=self.inv_v_n_i_s[::self.num_neg], prod=True)


        losses = cp.maximum(self.margin - wij + neg_wij.reshape(-1, self.num_neg).sum(axis=1) / self.num_neg, 0.)
        self.mask = cp.ones_like(losses)
        self.mask[cp.where(losses == 0.)] = 0.

        self.loss = losses.sum()

        return self.loss

    #TODO: rename variable neg_i, neg_j, etc.

    def m_grad(self, i, j, neg_i, neg_j, mask):
        xi, xj = self.means[i], self.c_means[j]
        neg_xi, neg_xj = self.means[neg_i], self.c_means[neg_j]
        pos = xj
        neg = neg_xj / self.num_neg

        # grad_i, grad_j, grad_nj
        return  ((- pos + neg.reshape(-1, self.num_neg, self.n_dim).sum(axis=1)) * mask.reshape(-1, 1)).reshape(-1, self.window_size, self.n_dim).sum(axis=1), \
                - xi * mask.reshape(-1, 1), \
                neg_xi * mask.repeat(self.num_neg).reshape(-1, 1) / self.num_neg

    def v_grad(self, i, j, neg_i, neg_j, vi, vj, neg_vi, neg_vj, v_i_s, v_n_i_s, inv_v_i_s, inv_v_n_i_s, mid_ij, mid_n_ij, inv_ij, inv_n_ij, mask):
        lvi, lvj = self.vars[i], self.c_vars[j]
        lneg_vi, lneg_vj = self.vars[neg_i], self.c_vars[neg_j]
        pos_i = wb.batch_log2(vi, vj, mid=mid_ij, inv_sU=inv_v_i_s, numIters=self.num_sqrt_iters, prod = True)
        neg_i_ = wb.batch_log2(neg_vi, neg_vj, mid=mid_n_ij, inv_sU=inv_v_n_i_s, numIters=self.num_sqrt_iters, prod=True) / self.num_neg
        pos_j = wb.batch_log(vj, vi, sV=v_i_s, inv=inv_ij, numIters = self.num_sqrt_iters, prod = True)
        neg_j_ = wb.batch_log(neg_vj, neg_vi, sV=v_n_i_s, inv=inv_n_ij, numIters=self.num_sqrt_iters, prod = True)

        return ((- (cp.matmul(pos_i, lvi)) + (cp.matmul(neg_i_, lneg_vi)).reshape(-1, self.num_neg, self.n_dim, self.n_dim).sum(axis=1)) * mask.reshape(-1, 1, 1)).reshape(-1, self.window_size, self.n_dim, self.n_dim).sum(axis=1), \
                 - (cp.matmul(pos_j, lvj)) * mask.reshape(-1, 1, 1), \
                (cp.matmul(neg_j_, lneg_vj)) * mask.repeat(self.num_neg).reshape(-1, 1, 1) / self.num_neg #+ 2 * self.var_reg * lneg_vj


    def SGD_update(self, i, j, neg_i, neg_j):

    #TODO : first function computes loss and gradient, second updates

        # Means gradients
        m_grad_i, m_grad_j, m_n_grad_j = self.m_grad(i, j, neg_i, neg_j, mask=self.mask)

        # Variances gradient
        v_grad_i, v_grad_j, v_n_grad_j = self.v_grad(i, j, neg_i, neg_j, vi=self.vi, vj=self.vj, neg_vi = self.neg_vi, neg_vj=self.neg_vj,
                                                     v_i_s = self.v_i_s, v_n_i_s = self.v_n_i_s, inv_v_i_s = self.inv_v_i_s,
                                                     inv_v_n_i_s = self.inv_v_n_i_s, mid_ij = self.mid_ij, mid_n_ij = self.mid_n_ij,
                                                     inv_ij = self.inv_ij, inv_n_ij = self.inv_n_ij, mask=self.mask)


        m_grad_i_acc, v_grad_i_acc, i_idxs = wb.sum_by_group(m_grad_i, v_grad_i, i)
        m_grad_j_acc, v_grad_j_acc, j_idxs = wb.sum_by_group(cp.concatenate([m_grad_j, m_n_grad_j]), cp.concatenate([v_grad_j, v_n_grad_j]), cp.concatenate([j, neg_j]))


        if self.optim == 'adagrad':

            self.means_adagrad[i_idxs] += m_grad_i_acc**2
            self.c_means_adagrad[j_idxs] += m_grad_j_acc**2
            self.vars_adagrad[i_idxs] += v_grad_i_acc**2
            self.c_vars_adagrad[j_idxs] += v_grad_j_acc**2

            # Means updates
            self.means[i_idxs] -= self.lr * m_grad_i_acc / cp.sqrt(self.means_adagrad[i_idxs] + self.epsilon)
            self.c_means[j_idxs] -= self.lr * m_grad_j_acc / cp.sqrt(self.c_means_adagrad[j_idxs] + self.epsilon)

            self.vars[i_idxs] -= self.lr * self.Cn * v_grad_i_acc / cp.sqrt(self.vars_adagrad[i_idxs] + self.epsilon)
            self.c_vars[j_idxs] -= self.lr * self.Cn * v_grad_j_acc / cp.sqrt(self.c_vars_adagrad[j_idxs] + self.epsilon)

        else:

            self.means[i_idxs] -= self.lr * m_grad_i_acc
            self.c_means[j_idxs] -= self.lr * m_grad_j_acc

            self.vars[i_idxs] -= self.lr * self.Cn * v_grad_i_acc
            self.c_vars[j_idxs] -= self.lr * self.Cn * v_grad_j_acc

        if self.unknown_words:
            self.means[0] *= 0
            self.c_means[0] *= 0
            self.vars[0] *= 0
            self.c_vars[0] *= 0


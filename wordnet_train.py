#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:35:41 2018

@author: boris
"""

import logging
import time

import cupy as cp
from cupy.cuda import Device

import utils as wb
from product import EllProduct
from softmax import EllSoftmax
from wordnet_data import Options



import multiprocessing


import pickle as pkl

import os
import errno
import argparse

parser = argparse.ArgumentParser(description='Train a skipgram ellipse embedding model')

parser.add_argument('--data', '-d', type = str, help = 'the data to train the model')
parser.add_argument('--output', '-o', type = str,
                    help = 'folder where to save the results')
parser.add_argument('--type', type=str, default = 'product',
                    help='which loss function?')
parser.add_argument('--optim', type=str, default = 'rmsprop',
                    help='optimization algorithm')
parser.add_argument('--learning_rate', '-lr', type = float, dest = 'lr', default = 1E-2)
parser.add_argument('--final_lr', type = float, default = 1E-3)
parser.add_argument('--decay', type = float, default = None)
parser.add_argument('--margin', type=float, default=1E-1, help='margin in the hinge loss ("product" type only)')
parser.add_argument('--dim', type = int, dest = 'dim', default = 2, help = 'dimension of the embeddings')
parser.add_argument('--epoches', '-e', type = int, default = 10)
parser.add_argument('--batch_size', '-b', type = int, dest = 'batch_size', default = 1000)
parser.add_argument('--neg_samples', '-ng', type = int, default = 50)
parser.add_argument('--epsilon', type=float, default=1E-8, help='adagrad/rmsprop parameter')
parser.add_argument('--device', type=int, default=4, help='gpu to use')
parser.add_argument('--num_sqrt_iters', type=int, default=6)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--embedding_file', type=str, default=None)
parser.add_argument('--save_each', type=int, default=5)
parser.add_argument('--lbda', type=float, default=1E-2)
parser.add_argument('--cn', type=float, default=1,
                    help = 'the means to Bures coefficient for learning ellipse embeddings')
args = parser.parse_args()

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

Device(args.device).use()

logging.info("Build dataset")

try:
    os.makedirs(args.output)
    os.makedirs(os.path.join(args.output, "embeddings"))
    os.makedirs(os.path.join(args.output, "figs"))
    os.makedirs(os.path.join(args.output, "losses"))
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

data = Options(args.data, save_path=os.path.join(args.output,'embeddings/vocab.txt'))

n_points = len(data.vocabulary)

if args.type == 'softmax':
    w_model = EllSoftmax(n_points, args.dim, lr=args.lr, num_neg=args.neg_samples, epsilon=args.epsilon,
                         window_size=1, optim = args.optim, scale = args.scale, lbda=args.lbda,
                         num_sqrt_iters=args.num_sqrt_iters,
                         Cn=args.cn, embedding_file=args.embedding_file, unknown_words = False,
                         sep_input_output=False)

if args.type == 'product':
    w_model = EllProduct(n_points, args.dim, lr=args.lr, num_neg=args.neg_samples,
                         window_size=1, optim = args.optim,
                         margin=args.margin, num_sqrt_iters=args.num_sqrt_iters,
                         Cn=args.cn, embedding_file=args.embedding_file, unknown_words = False,
                         sep_input_output=False)


with open(os.path.join(args.output, 'parameter_summary'), 'w') as summary_file:
    for arg in vars(args):
        summary_file.write(str(arg) + " : " + str(getattr(args, arg)) + "\n")


def worker(embedd_file, max_n=1000):
    """Rank and MAP evaluation worker"""
    os.system("python -W ignore wordnet_evaluation.py -e %s -d %s --device %u --max_n %u" % (embedd_file, args.data, (args.device + 1), max_n))
    return
logging.info("Start training")


if args.decay is not None:
    DECAY = args.decay
else :
    DECAY = (args.lr-args.final_lr) / args.epoches


#TODO: custom number of iterations between reports
for epoch in range(args.epoches): 
    total = 0
    moving_average = 0
    data.process = True
    start = time.time()
    n_iter = 1
    while data.process:
        batch_start = time.time()
        i, j, neg_i, neg_j = data.generate_batch(args.batch_size, num_neg=args.neg_samples)
        batch_end = time.time()
        step_start = time.time()
        loss = w_model.compute_loss(i, j, neg_i, neg_j)
        w_model.SGD_update(i, j, neg_i, neg_j)
        step_end = time.time()
        moving_average += loss
        if n_iter % (128 * 1000 / args.batch_size) == 0:
            with open(os.path.join(args.output,"losses/losses.txt"), 'a') as loss_file:
                loss_file.write('loss: %.6e \t batch time: %.6e \t compute time: %.6e \n'
                                % (moving_average / (128 * 1000), batch_end - batch_start, step_end - step_start))
            moving_average = 0
        n_iter += 1
        total += loss
    w_model.lr = max(args.final_lr, w_model.lr - DECAY)
    end = time.time()
    
    msg = 'Train Epoch: {} \tLoss: {} \t Time: {}'
    msg = msg.format(epoch, total, end-start)
    logging.info(msg)

    if epoch % args.save_each == 0:
        embedds = dict()
        embedds['means'] = cp.asnumpy(w_model.means)
        embedds['vars'] = cp.asnumpy(wb.to_full(
            w_model.vars) + args.lbda * cp.eye(args.dim).reshape(1, args.dim, args.dim).repeat(w_model.vars.shape[0],
                                                                                              axis=0))
        embedds['word_to_idx'] = data.word_to_index
        embedds['idx_to_word'] = data.vocabulary
        with open(os.path.join(args.output, "embeddings/embeddings" + '_' + str(epoch)), "wb") as output_file:
            pkl.dump(embedds, output_file)
        p = multiprocessing.Process(target=worker, args=(os.path.join(args.output, "embeddings/embeddings" + '_' + str(epoch)),))
        p.start()

    if w_model.optim == 'adagrad':
        w_model.means_adagrad *= 0
        w_model.c_means_adagrad *= 0
        w_model.vars_adagrad *= 0
        w_model.c_vars_adagrad *= 0

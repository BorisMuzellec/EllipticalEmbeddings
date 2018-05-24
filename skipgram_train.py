import logging
import time

import numpy as np
import cupy as cp

import gc

import copy

from softmax import EllSoftmax
from product import EllProduct

import utils as wr

from skipgram_data import Options

import pickle as pkl

import os
import errno

import argparse

from cupy.cuda import Device

parser = argparse.ArgumentParser(description='Train a skipgram ellipse embedding model')

parser.add_argument('--data', '-d', type=str, help='text files from which to train the model',
                    nargs='+', )
parser.add_argument('--output', '-o', type=str,
                    help='folder where to save the results')
parser.add_argument('--type', type=str, default = 'product',
                    help='which loss function?')
parser.add_argument('--optim', type=str, default = 'adagrad',
                    help='optimization algorithm')
parser.add_argument('--embedding_file', type=str, default=None)
parser.add_argument('--dim', type=int, dest='dim', default= 12, help='dimension of the embeddings')
parser.add_argument('--learning_rate', '-lr', type=float, dest='lr', default=1E-1)
parser.add_argument('--margin', type=float, default=10, help='margin in the hinge loss ("product" type only)')
parser.add_argument('--final_lr', type=float, default=1E-1)
parser.add_argument('--var_scale', type=float, default=1.0)
parser.add_argument('--lbda', type=float, default=1E-2)
parser.add_argument('--epoches', '-e', type=int, default=10)
parser.add_argument('--epsilon', type=float, default=1E-8, help = 'the initial value in RMSprop or adagrad')
parser.add_argument('--batch_size', '-b', type=int, dest='batch_size', default=10000)
parser.add_argument('--window_size', '-w', type=int, default=5)
parser.add_argument('--min_word_occ', '-mw', type=int, default=100)
parser.add_argument('--neg_samples', '-ng', type=int, default=1)
parser.add_argument('--chunk_size', '-cs', type=int, default=250000)
parser.add_argument('--num_sqrt_iters', type=int, default=6)
parser.add_argument('--cn', type=float, default=1,
                    help='the means to Bures coefficient for learning ellipse embeddings')
parser.add_argument('--chunks_dir', type=str, default='',
                    help='folder where to save the chunks')
parser.add_argument('--sample_dir', type=str, default='',
                    help='folder where to save the sampling files')
parser.add_argument('--load_from_files', action='store_true')
parser.add_argument('--read_from_chunks', action='store_true')
parser.add_argument('--no_sep_input_output', action='store_true')
parser.add_argument('--device', type=int, default=7)
args = parser.parse_args()

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

logging.info("Build dataset")

try:
    os.makedirs(args.output)
    os.makedirs(os.path.join(args.output, "embeddings"))
    os.makedirs(os.path.join(args.output, "figs"))
    os.makedirs(os.path.join(args.output, "losses"))
    os.makedirs(os.path.join(args.output, "norms"))
    os.makedirs(args.sample_dir)
    os.makedirs(args.chunks_dir)

except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

try:
    os.makedirs(args.sample_dir)

except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

if args.chunks_dir:
    try:
        os.makedirs(args.chunks_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

data = Options(args.data, min_occurences=args.min_word_occ,
               save_path=args.output, chunk_size=args.chunk_size,
               save_data=args.sample_dir, save_chunks=args.chunks_dir,
               load_from_files=args.load_from_files,
               read_from_chunks=args.read_from_chunks)

Device.use(Device(args.device))

if args.type == 'softmax':
    w_model = EllSoftmax(data.vocabulary_size, args.dim, lr=args.lr, num_neg=args.neg_samples, window_size=2 * args.window_size,
                         num_sqrt_iters = args.num_sqrt_iters, margin=args.margin, optim = args.optim,
                         Cn=args.cn, embedding_file=args.embedding_file)

if args.type == 'product':
    w_model = EllProduct(data.vocabulary_size, args.dim, lr=args.lr, num_neg=args.neg_samples, var_scale=args.var_scale,
                         window_size=2 * args.window_size, optim=args.optim, epsilon=args.epsilon,
                         margin=args.margin, num_sqrt_iters=args.num_sqrt_iters, lbda=args.lbda,
                         Cn=args.cn, embedding_file=args.embedding_file,
                         sep_input_output=not args.no_sep_input_output)


with open(os.path.join(args.output, 'parameter_summary'), 'w') as summary_file:
    for arg in vars(args):
        summary_file.write(str(arg) + " : " + str(getattr(args, arg)) + "\n")

logging.info("Start training")

DECAY = (args.lr - args.final_lr) / (args.epoches * 1E3 / args.batch_size)

for epoch in range(args.epoches):

    total = 0
    data.process = True
    start = time.time()
    niter = 1
    moving_average = 0

    while data.process:
        batch_start = time.time()
        #TODO: delete the zero words when generating batches ?
        i, j, neg_i, neg_j = data.generate_batch(args.window_size, args.batch_size, num_neg=args.neg_samples)
        i_ = i[i != 0]
        j_ = j[i != 0]
        neg_i_ = neg_i[neg_i != 0]
        neg_j_ = neg_j[neg_i != 0]
        i, j, neg_i, neg_j = i_, j_, neg_i_, neg_j_
        batch_end = time.time()
        step_start = time.time()

        loss = w_model.compute_loss(i, j, neg_i, neg_j)
        w_model.SGD_update(i, j, neg_i, neg_j)

        step_end = time.time()

        niter += 1
        total += loss
        moving_average += loss

        w_model.lr = max(w_model.lr - DECAY, args.final_lr)

        #TODO: report_each instead
        if niter % int(1E6 / args.batch_size) == 0:

            with open(os.path.join(args.output, "losses/losses.txt"), 'a') as loss_file:
                loss_file.write('loss: %.6e \t batch time: %.6e \t compute time: %.6e \n'
                                % (moving_average / (1E6 * 2 * args.window_size), batch_end - batch_start, step_end - step_start))
            moving_average = 0

    end = time.time()

    msg = 'Train Epoch: {} \tLoss: {} \t Time: {}'
    msg = msg.format(epoch, total, end - start)
    logging.info(msg)

    embedds = dict()
    embedds['means'] = cp.asnumpy(w_model.means)
    embedds['vars'] = cp.asnumpy(wr.to_full(w_model.vars) + args.lbda * cp.eye(args.dim).reshape(1, args.dim, args.dim).repeat(w_model.vars.shape[0], axis=0))
    embedds['vars'][0] *= 0
    embedds['c_means'] = cp.asnumpy(w_model.c_means)
    embedds['c_vars'] = cp.asnumpy(wr.to_full(w_model.c_vars) + args.lbda * cp.eye(args.dim).reshape(1, args.dim, args.dim).repeat(w_model.c_vars.shape[0], axis=0))
    embedds['c_vars'][0] *= 0

    with open(os.path.join(args.output, "embeddings/embeddings" + '_' + str(epoch + 1)), "wb") as output_file:
        pkl.dump(embedds, output_file)

    with open(os.path.join(args.output, "losses/epoches" + '_' + str(epoch + 1)), "wb") as epoch_file:
        epoch_file.write(str.encode('epoch: %u \t loss: %.6e\n' % (epoch + 1, total)))

    if w_model.optim == 'adagrad':
        w_model.means_adagrad *= 0
        w_model.c_means_adagrad *= 0
        w_model.vars_adagrad *= 0
        w_model.c_vars_adagrad *= 0


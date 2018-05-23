#cython: boundscheck=False, wraparound=False, embedsignature=True, cdivision = True

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:49:04 2018

@author: boris
"""

cimport cython

cimport numpy as np
import numpy as np
import cupy as cp


import pandas as pd
import pickle as pkl
import os
import copy
from cy_sample import cython_get_sample

from collections import Counter
import random

#from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin


cdef class Options(object):

  cdef int data_index
  cdef int sample_n
  cdef char* save_path, datafile
  cdef public np.ndarray train_data, sample_table, _node_probabilities
  cdef public np.ndarray vocabulary
  cdef public dict word_to_index, relations_dict
  cpdef public bint process
  cpdef set indices_set
  cdef public long vocabulary_size, table_size
  cpdef public bint undirected
    
  def __cinit__(self, datafile, char* save_path = 'data', bint undirected = True):
      self.save_path = save_path
  
      self.vocabulary, data_or = self.read_data(datafile)
      self.undirected = undirected
      self.word_to_index, self.train_data, self.relations_dict = self.build_dataset(data_or, self.undirected)

      self.vocabulary_size = len(self.vocabulary)
      self.indices_set = set(range(self.vocabulary_size))

      # if not os.path.isfile('wnet_sample_table.pkl'):
      #   self.sample_table = self.init_sample_table()
      #   with open('wnet_sample_table.pkl', 'w') as sample_file:
      #       pkl.dump(self.sample_table, sample_file)
      # else:
      #   self.sample_table = pkl.loadopen('wnet_sample_table.pkl', 'rb')

      self.sample_table = self.init_sample_table()
      self.table_size = len(self.sample_table)

      prob_dict = {k: float(v) / self.table_size for k, v in Counter(self.sample_table).items()}
      self._node_probabilities = np.array([prob_dict[i] for i in range(self.vocabulary_size)])

      np.random.shuffle(self.train_data)
      np.random.shuffle(self.sample_table)

      self.process = True
      self.data_index = 0
      self.sample_n = 0


    
      #Release some RAM
      del data_or

    
  def read_data(self, char* filename):
      df = pd.read_csv(filename, sep = '\t', header = None)
      # df[0] = df[0].apply(lambda x: x.split('.')[0])
      # df[1] = df[1].apply(lambda x: x.split('.')[0])
      return (df[0].append(df[1])).unique(), df.get_values()

  cpdef tuple build_dataset(self, data_or, undirected):
      """Process raw inputs into a ."""
      cpdef dict dictionary = {}
      cpdef dict relations_dict = {}
      cdef size_t i
      cdef str item
      cdef list item_relations
      cdef int n_words = len(self.vocabulary)
      cpdef int data_length
      data_length = data_or.shape[0] if undirected else 2 * data_or.shape[0]
      cdef np.ndarray[np.int_t, ndim = 2] data = np.empty((data_length, data_or.shape[1]), dtype = np.int)
    
      for i in range(n_words):
          dictionary[self.vocabulary[i]] = i
          relations_dict[i] = set()
      
      for i in range(len(data_or)):
          data[i][0] = dictionary[data_or[i][0]]
          data[i][1] = dictionary[data_or[i][1]]
          relations_dict[data[i][0]].add(data[i][1])
          relations_dict[data[i][1]].add(data[i][0])

       #Same data with inversed roles (for symmetry)
      if not undirected:
          for i in range(len(data_or), 2*len(data_or)):
              data[i][0] = dictionary[data_or[i - len(data_or)][1]]
              data[i][1] = dictionary[data_or[i - len(data_or)][0]]

      # for i in range(n_words):
      #     item_relations = list(set(df[df[0] == self.vocabulary[i]][1]).union(set(df[df[1] == self.vocabulary[i]][0])))
      #     relations_dict[i] = [dictionary[item] for item in item_relations]
        
      return dictionary, data, relations_dict

  cpdef save_vocab(self):
    cpdef dict maps = dict()
    maps['idx_to_word'] = self.vocabulary
    maps['word_to_idx'] = self.word_to_index
    #maps['relation_dict'] = self.relations_dict
    with open(self.save_path, "w") as f:
        pkl.dump(maps, f)
        
  cpdef np.ndarray init_sample_table(self):
      cdef size_t i
      # cpdef dict sample_table = {}
      #return np.arange(len(self.vocabulary))
      return np.array([self.train_data[i][0] for i in range(len(self.train_data))] + [self.train_data[i][1] for i in range(len(self.train_data))])
      # for i in range(self.vocabulary_size):
      #     sample_table[i] = np.array(list(self.indices_set - self.relations_dict[i]))

      return sample_table


  cpdef tuple generate_batch(self, int batch_size, int num_neg, bint cuda = True):
    cdef np.ndarray[np.int_t, ndim = 2] data
    cdef list neg_samples
    cpdef np.ndarray[np.int_t, ndim=1] pos_u = np.empty(batch_size, dtype=np.int)
    cpdef np.ndarray[np.int_t, ndim=1] pos_v = np.empty(batch_size, dtype=np.int)
    cpdef np.ndarray[np.int_t, ndim=1] neg_u = np.empty(num_neg * batch_size, dtype=np.int)
    cpdef np.ndarray[np.int_t, ndim=1] neg_v = np.empty(num_neg * batch_size, dtype=np.int)
    cpdef size_t i, j
    
    data = self.train_data

    for i in range(batch_size):
        pos_u[i] = data[(self.data_index + i) % len(data)][0]
        pos_v[i] = data[(self.data_index + i) % len(data)][1]
        # neg_samples = self._sample_negatives(pos_u[i], num_neg)
        for j in range(num_neg):
            neg_u[i*num_neg + j] = pos_u[i]
            # neg_v[i*num_neg + j] = neg_samples[j]

            
    self.data_index += batch_size
            
    if self.data_index + batch_size > len(data):
        np.random.shuffle(self.train_data)
        self.data_index = 0
        self.sample_n = 0
        self.process = False 
        
#    neg_v = get_sample(self.sample_table, arr_len = self.table_size, n_iter = self.sample_n,
#                              sample_size = num_neg * batch_size, fast = True)
    neg_v = self.sample_table[np.random.randint(self.table_size, size=num_neg * batch_size)]
    neg_v = self.resample_positives(neg_v, neg_u)
    
    # self.sample_n += 1

    return cp.array(pos_u), cp.array(pos_v), cp.array(neg_u), cp.array(neg_v)

  cdef list _sample_negatives(self, long node_index, int num_neg):
    """Return a sample of negatives for the given node.
    Parameters
    ----------
    node_index : int
        Index of the positive node for which negative samples are to be returned.
    Returns
    -------
    numpy.array
        Array of shape (self.negative,) containing indices of negative nodes for the given node index.
    """

    cpdef set node_relations, unique_indices
    cpdef np.ndarray indices, valid_negatives, probs
    cpdef int num_remaining_nodes

    node_relations = self.relations_dict[node_index]
    num_remaining_nodes = self.vocabulary_size - len(node_relations)

    if num_remaining_nodes < num_neg:
            raise ValueError(
                'Cannot sample %d negative nodes from a set of %d negative nodes for %s' %
                (self.negative, num_remaining_nodes, self.kv.index2word[node_index])
            )

    positive_fraction = float(len(node_relations)) / self.vocabulary_size
    if positive_fraction < 1:
        # If number of positive relations is a small fraction of total nodes
        # re-sample till no positively connected nodes are chosen
        # indices = np.random.choice(self.sample_table, replace=False, size=num_neg)
        indices = self.sample_table[np.random.randint(self.table_size, size=num_neg)]
        unique_indices = set(indices)
        times_sampled = 1
        while (unique_indices & node_relations):
            times_sampled += 1
            indices = self.sample_table[np.random.randint(self.table_size, size=num_neg)]   
            unique_indices = set(indices)
        # if times_sampled > 1:
            # print('Sampled %d times, positive fraction %.5f', times_sampled, positive_fraction)
    else:
        print("Sampling positives only")
        # If number of positive relations is a significant fraction of total nodes
        # subtract positively connected nodes from set of choices and sample from the remaining
        valid_negatives = np.array(list(self.indices_set - node_relations))
        probs = self._node_probabilities[valid_negatives]
        probs /= probs.sum()
        indices = np.random.choice(valid_negatives, size=num_neg, p=probs, replace=False)

    return list(indices)

  cdef np.ndarray resample_positives(self, np.ndarray neg_v, np.ndarray neg_u):
      cdef size_t i
      cdef set node_relations
      cdef int num_remaining_nodes

      for i in range(len(neg_v)):
          node_relations = self.relations_dict[neg_u[i]]
          num_remaining_nodes = self.vocabulary_size - len(node_relations)
          if num_remaining_nodes <= 1:
            neg_v[i] =  neg_u[i]
          else:
            while set([neg_v[i]]) & node_relations:
                neg_v[i] = self.sample_table[np.random.randint(self.table_size)]

      return neg_v




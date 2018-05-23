#cython: boundscheck=False, embedsignature=True

#!/usr/bin/env python2
# -*- coding: utf-8 -*-


cimport cython
from libc.math cimport sqrt
import logging

import tqdm

from collections import Counter
cimport numpy as np
import numpy as np

import string

import cupy as cp

import pickle as pkl
import os

from itertools import islice

from sampling import get_sample
import random


from six.moves import xrange


cdef class Options(object):

  cdef int data_index
  cdef int sample_n
  cpdef int min_occurences
  cpdef public int vocabulary_size
  cdef int current_file_idx
  cdef int chunk_size
  cpdef public str save_path
  cpdef public str save_data
  cpdef public str save_chunks
  cdef np.ndarray sample_table
  cdef public dict vocab_words
  cdef list count, current_train_data, chunk_files, train_files
  cpdef public bint process, load_from_files, read_from_chunks
  cdef long n_vocabulary, table_size
  cdef float thresh
    
  def __cinit__(self, list datafiles, int min_occurences, str save_path = 'data', thresh=1E-5,
                str save_data = 'data', str save_chunks = 'chunks', int chunk_size = 10000,
                bint load_from_files = False, bint read_from_chunks = False):
    self.chunk_size = chunk_size
    self.min_occurences = min_occurences
    self.save_path = save_path
    self.save_data = save_data
    self.save_chunks = save_chunks
    self.thresh = thresh

    if load_from_files:
        self.build_from_files(self.save_data)
    
    else:
        if read_from_chunks:
            count_dict, chunk_files = self.read_data_from_chunks()    
        else:
            count_dict, chunk_files = self.read_data(datafiles)
        
        self.train_files, self.count, self.vocab_words = self.build_dataset(count_dict, 
                                                                            self.min_occurences,
                                                                            chunk_files)

    self.vocabulary_size = len(self.count)
    np.random.shuffle(self.train_files)
    
    #Release some RAM
    if 'count_dict' in locals():
        del count_dict
        del chunk_files

    if not os.path.isfile((os.path.join(self.save_data, 'sample_table.pkl'))):
        self.sample_table = self.init_sample_table()
    else:
        self.sample_table = pkl.load(open(os.path.join(self.save_data, 'sample_table.pkl'), 'r'))

    self.table_size = len(self.sample_table)
    np.random.shuffle(self.sample_table) #allows efficient sampling

    self.process = True
    self.data_index = 0
    self.sample_n = 0
    self.current_file_idx = 0
    with open(os.path.join(self.save_data, self.train_files[0]), "r") as train_file:
        self.current_train_data = pkl.load(train_file)
        
  cpdef void build_from_files(self, str datadir):

      cdef str file
      self.train_files = list()
      
      for file in os.listdir(datadir):
          if file.split('_')[-2] == 'sampling':
              self.train_files.append(file)
      
      with open(os.path.join(datadir, 'count_dict.pkl'), "r") as countfile:
          self.count = pkl.load(countfile)

      with open(os.path.join(datadir, 'vocab_words.pkl'), "r") as vocabfile:
          self.vocab_words = pkl.load(vocabfile)
    
  cpdef tuple read_data(self, list filenames):

    cpdef size_t i = 0
    cpdef list chunk_files = []
    cpdef list chunk
    cpdef list lines
    cpdef str filename
    count = Counter() 
    
    for filename in filenames:
        i = 0
        
        with open(filename, 'r') as f:
            
            while True:
                lines = list(islice(f, self.chunk_size))
                if not lines:
                    break
                
                chunk = (' '.join([line for line in lines if not line.startswith('CURRENT URL')])).translate(None, string.punctuation).lower().split()
                count.update(chunk)
                
                with open(os.path.join(self.save_chunks, filename.split('/')[-1] + "_chunk_" + str(i)), 'w') as chunk_file:
                    pkl.dump(chunk, chunk_file)
                    chunk_files.append(filename.split('/')[-1] + "_chunk_" + str(i))
                    
                i += 1
        
    print("Done chunkifying")
    return count, chunk_files
    
  cpdef tuple read_data_from_chunks(self):

    cpdef list chunk_files = []
    cpdef list data
    cpdef str filename
    count = Counter() 
    
    for filename in tqdm.tqdm(os.listdir(self.save_chunks)):
        
        if filename.split('_')[-2] == 'chunk':
            with open(os.path.join(self.save_chunks, filename), 'r') as f:
                data = pkl.load(f)
                count.update(data)
                chunk_files.append(filename)
            
    print("Done chunkifying")
    return count, chunk_files

  cpdef tuple build_dataset(self, count_dict, int min_occurences, list chunk_files):
    """Process raw inputs into a ."""
 
    cpdef list count
    cpdef list sample_files
    cpdef list data
    cpdef list chunk
    cpdef dict dictionary = {}
    cpdef dict reversed_dictionary = {}
    cdef char* word
    cdef long c
    cdef long index

    count = [['UNK', -1]]
    count_dict = Counter({i:count_dict[i] for i in count_dict if count_dict[i] >= min_occurences})
    count.extend(count_dict.most_common())
    count[0][1] = sum(count_dict.values()) - sum([ele[1] for ele in count])

    cpdef int n_words = len(count)
    
    for i in xrange(n_words):
      dictionary[count[i][0]] = len(dictionary)
    
    cpdef np.ndarray[double, ndim=1] frequency
    cpdef np.ndarray[long, ndim = 1] counts
    cpdef dict P = {}
    cdef double x
    cdef double y
    cdef char* filename
      
    counts = np.array([ele[1] for ele in count])
    frequency = counts / float(sum(counts))
    
    for idx in xrange(len(frequency)):
      x = frequency[idx]
      y = (sqrt(x/self.thresh)+1)*self.thresh/x
      P[idx] = y
    
    sample_files = []
    
    for filename in chunk_files:
        
        data = list()
        
        with open(os.path.join(self.save_chunks, filename), 'r') as file:
            chunk = pkl.load(file)
            
        for i in xrange(len(chunk)):
            if chunk[i] in dictionary:
                index = dictionary[chunk[i]]
            else:
                index = 0  # dictionary['UNK']
            if random.random()<P[index]:
                data.append(index)
          
        with open(os.path.join(self.save_data, '_'.join(filename.split('_')[:-2]) + "_sampling_" + filename.split('_')[-1]), 'w') as sample_file:
            pkl.dump(data, sample_file)
            sample_files.append('_'.join(filename.split('_')[:-2]) + "_sampling_" + filename.split('_')[-1])
    
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    #Save everything
    
    with open(os.path.join(self.save_data, 'count_dict.pkl'), 'w') as countfile:
          pkl.dump(count, countfile)

    with open(os.path.join(self.save_data, 'vocab_words.pkl'), 'w') as vocab_words:
          pkl.dump(reversed_dictionary, vocab_words)

    with open(os.path.join(self.save_data, 'words_to_idxs.pkl'), 'w') as dictfile:
          pkl.dump(dictionary, dictfile)
          
    return sample_files, count, reversed_dictionary
    

  cpdef np.ndarray[unsigned long, ndim = 1] init_sample_table(self):
    cpdef np.ndarray[long, ndim = 1] count
    cpdef np.ndarray[double, ndim = 1] ratio_count
    cpdef np.ndarray[double, ndim = 1] pow_frequency
    cpdef list sample_table = []
    cpdef size_t idx
    cdef long table_size = int(1e8)
    
    count = np.array([ele[1] for ele in self.count])
    pow_frequency = count**0.75
    power = sum(pow_frequency)
    ratio = pow_frequency/float(power)
    ratio_count = np.round(ratio*table_size)

    for idx in xrange(1, len(ratio_count)):
      sample_table += [idx]*int(ratio_count[idx])

    with open(os.path.join(self.save_data, 'sample_table.pkl'), 'w') as sample_file:
      pkl.dump(np.array(sample_table), sample_file)

    return np.array(sample_table)
    
  cpdef np.ndarray[double, ndim = 1] weight_table(self):
    cpdef np.ndarray[long, ndim = 1] count
    cpdef np.ndarray[double, ndim=1] pow_frequency
    count = np.array([ele[1] for ele in self.count])
    pow_frequency = count**0.75
    power = sum(pow_frequency)
    ratio = pow_frequency / float(power)
    return ratio

  cpdef tuple generate_batch(self, int window_size, int batch_size, int num_neg, bint cuda = True):
    cdef list data
    cpdef int span = 2 * window_size + 1
    cpdef np.ndarray[long, ndim = 1] neg_v, labels
    cpdef np.ndarray[long, ndim = 2] context
    cpdef np.ndarray[np.int_t, ndim=1] pos_u = np.empty((span - 1) * batch_size, dtype=np.int)
    cpdef np.ndarray[np.int_t, ndim=1] pos_v = np.empty((span - 1) * batch_size, dtype=np.int)
    cpdef np.ndarray[np.int_t, ndim=1] neg_u = np.empty((span - 1) * num_neg * batch_size, dtype=np.int)
    cpdef np.ndarray[np.int_t, ndim=1] buffer = np.empty(span, dtype = np.int)
    cpdef size_t i, j, k
    
    data = self.current_train_data
    context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
    labels = np.ndarray(shape=(batch_size), dtype=np.int64)
    
    if self.data_index + span > len(data):
      self.data_index = 0
      self.current_file_idx += 1
      
      if self.current_file_idx > len(self.train_files) - 1:
          self.process = False
          self.sample_n = 0
          self.current_file_idx = 0
                
      with open(os.path.join(self.save_data,self.train_files[self.current_file_idx]), "r") as train_file:
        self.current_train_data = pkl.load(train_file)
      
    for i in range(span):
        buffer[i] = data[self.data_index + i]

    for i in range(batch_size):
        self.data_index += 1
        for j in range(window_size):
            context[i,j] = buffer[j]
        for j in range(window_size, 2 * window_size):
            context[i,j] = buffer[j + 1]
        labels[i] = buffer[window_size]
        if self.data_index + span > len(data):
            #Load a new file and stop this batch now
            self.data_index = 0
            self.current_file_idx += 1

            #Open a new file
            if self.current_file_idx > len(self.train_files) - 1:
                self.process = False
                self.sample_n = 0
                np.random.shuffle(self.train_files)
                self.current_file_idx = 0

            with open(os.path.join(self.save_data,self.train_files[self.current_file_idx]), "r") as train_file:
                logging.info("Opening %s \n" % os.path.join(self.save_data,self.train_files[self.current_file_idx]))
                self.current_train_data = pkl.load(train_file)

            data = self.current_train_data

            pos_u = pos_u[:i * (span-1)]
            pos_v = pos_v[:i * (span-1)]
            neg_u = neg_u[:i * (span-1) * num_neg]
            neg_v = get_sample(self.sample_table, arr_len = self.table_size, n_iter = self.sample_n,
                          sample_size = len(neg_u), fast = True)

            self.sample_n += 1

            if cuda:
                return cp.array(pos_u), cp.array(pos_v), cp.array(neg_u), cp.array(neg_v)
            else:
                return pos_u, pos_v, neg_u, neg_v
        
        else:
            for j in range(span):
                buffer[j] = data[self.data_index + j]

        for j in range(span-1):
            pos_u[i*(span-1) + j] = labels[i]
            for k in range(num_neg):
                neg_u[(i*(span-1) + j) * num_neg + k] = labels[i]
            pos_v[i*(span-1) + j] = context[i,j]
        
    neg_v = get_sample(self.sample_table, arr_len = self.table_size, n_iter = self.sample_n,
                              sample_size = num_neg * batch_size * 2 * window_size, fast = True)
    
    self.sample_n += 1
    
    if cuda:
        return cp.array(pos_u), cp.array(pos_v), cp.array(neg_u), cp.array(neg_v)
    else:
        return pos_u, pos_v, neg_u, neg_v



        
import numpy as np
cimport numpy as np
cimport cython # so we can use cython decorators
from cpython cimport bool # type annotation for boolean

np.import_array()

# disable index bounds checking and negative indexing for speedups
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef get_sample(np.ndarray arr, int arr_len, int n_iter, int sample_size,
                 bool fast):
    cdef int start_idx
    if fast:
        start_idx = (n_iter * sample_size) % arr_len
        if start_idx + sample_size >= arr_len:
            np.random.shuffle(arr)
            start_idx = 0
            
        return arr[start_idx:start_idx+sample_size] 
    else:
        return np.random.choice(arr, sample_size, replace=False)

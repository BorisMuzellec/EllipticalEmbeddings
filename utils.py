#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:23:43 2018

@author: boris
"""

import numpy as np
import cupy as cp

def wishart(n_points, dim=2, p=5):
    """
    Wishart sampling
    """
    X = cp.random.randn(n_points, dim, p)
    return cp.matmul(X, cp.transpose(X, axes=[0, 2, 1]))


def batch_sqrtm(A, numIters = 20, reg = 2.0):
    """
    Batch matrix root via Newton-Schulz iterations
    """

    batchSize = A.shape[0]
    dim = A.shape[1]
    #Renormalize so that the each matrix has a norm lesser than 1/reg, but only normalize when necessary
    normA = reg * cp.linalg.norm(A, axis=(1, 2))
    renorm_factor = cp.ones_like(normA)
    renorm_factor[cp.where(normA > 1.0)] = normA[cp.where(normA > 1.0)]
    renorm_factor = renorm_factor.reshape(batchSize, 1, 1)

    Y = cp.divide(A, renorm_factor)
    I = cp.eye(dim).reshape(1, dim, dim).repeat(batchSize, axis=0)
    Z = cp.eye(dim).reshape(1, dim, dim).repeat(batchSize, axis=0)
    for i in range(numIters):
        T = 0.5 * (3.0 * I - cp.matmul(Z, Y))
        Y = cp.matmul(Y, T)
        Z = cp.matmul(T, Z)
    sA = Y * cp.sqrt(renorm_factor)
    sAinv = Z / cp.sqrt(renorm_factor)
    return sA, sAinv


def batch_bures(U, V, numIters = 20, U_stride=None, sU = None, inv_sU = None, prod = False):
    #Avoid recomputing roots if not necessary
    if sU is None:
        if U_stride is not None:
            sU_, inv_sU_ = batch_sqrtm(U[::U_stride], numIters=numIters)
            sU = sU_.repeat(U_stride, axis=0)
            inv_sU = inv_sU_.repeat(U_stride, axis=0)
        else :
            sU, inv_sU = batch_sqrtm(U, numIters=numIters)
    cross, inv = batch_sqrtm(cp.matmul(sU, cp.matmul(V, sU)), numIters = numIters)
    if prod:
        return cp.trace(cross, axis1=1, axis2=2), inv, sU, inv_sU, cross
    else:
        return cp.trace(U + V - 2 * cross, axis1=1, axis2=2), inv, sU, inv_sU, cross


def batch_W2(m1, m2, U, V, Cn = 1, numIters = 20, U_stride=None, sU = None, inv_sU = None, prod = False):
    """
    Squared Wasserstein distance between N(m1, U) and N(m2, V)
    """
    bb, inv, sU, inv_sU, mid = batch_bures(U, V, numIters = numIters, U_stride=U_stride, sU = sU, inv_sU = inv_sU, prod=prod)
    if prod:
        return (m1*m2).sum(axis=1) + Cn * bb, inv, sU, inv_sU, mid
    else:
        return ((m1 - m2)**2).sum(axis=1) + Cn * bb, inv, sU, inv_sU, mid
    
def batch_Tuv(U, V, inv=None, sV=None, numIters = 2):
    """
    Returns the transportation matrix from N(U) to N(V):
    V^{1/2}[V^{1/2}UV^{1/2}]^{-1/2}V^{1/2}
    """
    if sV is None:
        sV, _ = batch_sqrtm(V, numIters=numIters)
    if inv is None:
        _, inv = batch_sqrtm(cp.matmul(cp.matmul(sV, U), sV), numIters = numIters)
    return cp.matmul(sV, cp.matmul(inv, sV))

def batch_log(U, V, inv=None, sV=None, numIters = 2, prod = False):
    """
    Log map at N(U) of N(V)
    """
    batchsize = U.shape[0]
    n = U.shape[1]
    if prod:
        return batch_Tuv(U, V, inv, sV, numIters=numIters)
    else:
        return batch_Tuv(U, V, inv, sV, numIters = numIters) - cp.eye(n).reshape(1, n, n).repeat(batchsize, axis=0)

def batch_Tuv2(U, V, mid=None, inv_sU=None, numIters = 2):
    """
    Returns the transportation matrix from N(U) to N(V):
    V^{-1/2}[V^{1/2}UV^{1/2}]^{1/2}V^{-1/2}
    """
    if (inv_sU is None) or (mid is None):
        sU, inv_sU = batch_sqrtm(U, numIters = numIters)
    if mid is None:
        mid, _ = batch_sqrtm(cp.matmul(cp.matmul(sU, V), sU), numIters = numIters)
    return cp.matmul(inv_sU, cp.matmul(mid, inv_sU))

def batch_log2(U, V, mid=None, inv_sU=None, numIters = 2, prod = False):
    """
    Log map at N(U) of N(V)
    """
    batchsize = U.shape[0]
    n = U.shape[1]
    if prod:
        return batch_Tuv2(U, V, mid, inv_sU, numIters=numIters)
    else:
        return batch_Tuv2(U, V, mid, inv_sU, numIters = numIters) - cp.eye(n).reshape(1, n, n).repeat(batchsize, axis=0)
    
def batch_exp(U, V):
    """
    Exponential map at N(U) in the direction of V
    """
    batchsize = U.shape[0]
    n = V.shape[1]
    V_I = V + cp.eye(n).reshape(1, n, n).repeat(batchsize, axis=0)
    return cp.matmul(V_I, cp.matmul(U, V_I))


def to_full(L):
    xp = cp.get_array_module(L)
    return xp.matmul(L, xp.transpose(L, axes=(0, 2, 1)))

def diag_bures(U, V):
    """
    Batched squared bures distance between diagonal covariances U and V, represented as batch of vectors
    """
    return ((cp.sqrt(U) - cp.sqrt(V))**2).sum(axis=1)

def diag_W2(m1, m2, U, V, Cn = 1):
    """
    Squared Wasserstein distance between E(m1, U) and E(m2, V), where U and V are diagonal matrices
    """
    return ((m1 - m2)**2).sum(axis=1) + Cn * diag_bures(U, V)

def bures_cosine(m1, m2, U, V, Cn = 1, numIters = 20):
    bb = batch_bures(U, V, numIters = numIters, prod=True)[0]

    return ((m1*m2).sum(axis=1) + Cn * bb) / cp.sqrt((cp.linalg.norm(m1, axis=1)**2 + Cn * cp.trace(U, axis1=1, axis2=2) + 1E-8) * (cp.linalg.norm(m2, axis=1)**2 + Cn * cp.trace(V, axis1=1, axis2=2) + 1E-8))


def sum_cosine(m1, m2, U, V, Cn = 1, numIters = 20):

    bb = batch_bures(U, V, numIters = numIters, prod=True)[0]
    #print cp.mean(bb)
    #print cp.mean((m1*m2).sum(axis=1))

    return (m1*m2).sum(axis=1) / (cp.linalg.norm(m1, axis=1) * cp.linalg.norm(m2, axis=1) + 1E-8)\
           + Cn * bb / cp.sqrt((cp.trace(U, axis1=1, axis2=2) * cp.trace(V, axis1=1, axis2=2)) + 1E-8)

def sum_by_group(values1, values2, groups):
    order = cp.argsort(groups)
    groups = groups[order]
    values1 = values1[order]
    values2 = values2[order]
    cp.cumsum(values1, out=values1, axis=0)
    cp.cumsum(values2, out=values2, axis=0)
    index = cp.ones(len(groups), 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values1 = values1[index]
    values2 = values2[index]
    groups = groups[index]
    values1[1:] = values1[1:] - values1[:-1]
    values2[1:] = values2[1:] - values2[:-1]
    return values1, values2, groups


def symmetrize(M):
    return (M + cp.transpose(M, axes=(0, 2, 1))) / 2.0
import random
import logging
import time
#import gnumpy as gnp
import numpy as np
# logging.basicConfig(level=logging.DEBUG)
import multiprocessing
import sys
import os
#import torch
#from scipy import sparse

from scipy.sparse.linalg import svds

def renorm(X):
	Y = X.copy()
	Y = Y.astype(float)
	ngene,nsample = Y.shape
	s = np.sum(Y, axis=0)
	#print s.shape()
	for i in range(nsample):
		if s[i]==0:
			s[i] = 1
			if i < ngene:
				Y[i,i] = 1
			else:
				for j in range(ngene):
					Y[j,i] = 1. / ngene
		Y[:,i] = Y[:,i]/s[i]
	return Y

def RandomWalkRestart(A, rst_prob, delta = 1e-4, reset=None, max_iter=50,use_torch=False,return_torch=False):
	if use_torch:
		device = torch.device("cuda:0")
	nnode = A.shape[0]
	#print nnode
	if reset is None:
		reset = np.eye(nnode)
	nsample,nnode = reset.shape
	#print nsample,nnode
	P = renorm(A)
	P = P.T
	norm_reset = renorm(reset.T)
	norm_reset = norm_reset.T
	if use_torch:
		norm_reset = torch.from_numpy(norm_reset).float().to(device)
		P = torch.from_numpy(P).float().to(device)
	Q = norm_reset

	for i in range(1,max_iter):
		#Q = gnp.garray(Q)
		#P = gnp.garray(P)
		if use_torch:
			Q_new = rst_prob*norm_reset + (1-rst_prob) * torch.mm(Q, P)#.as_numpy_array()
			delta = torch.norm(Q-Q_new, 2)
		else:
			Q_new = rst_prob*norm_reset + (1-rst_prob) * np.dot(Q, P)#.as_numpy_array()
			delta = np.linalg.norm(Q-Q_new, 'fro')
		Q = Q_new
		#print 'random walk iter',i, delta
		sys.stdout.flush()
		if delta < 1e-4:
			break
	if use_torch and not return_torch:
		Q = Q.cpu().numpy()
	return Q

def DCA_vector(Q, dim):
	nnode = Q.shape[0]
	alpha = 1. / (nnode **2)
	Q = np.log(Q + alpha) - np.log(alpha);

	#Q = Q * Q';
	[U, S, V] = svds(Q, dim);
	S = np.diag(S)
	X = np.dot(U, np.sqrt(S))
	Y = np.dot(np.sqrt(S), V)
	Y = np.transpose(Y)
	return X,U,S,V,Y

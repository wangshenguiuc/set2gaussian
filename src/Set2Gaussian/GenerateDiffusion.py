import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import collections
import operator
import pickle
import sys
import random
import os
from scipy import sparse
from scipy import spatial
from sklearn import metrics
from sklearn.svm import SVC
import cPickle as pickle
from sklearn.model_selection import GridSearchCV,KFold
from BioNetwork import BioNetwork
import scipy.spatial as sp
from sklearn.decomposition import SparsePCA
import time
from sklearn.cross_validation import train_test_split

class GenerateDiffusion:
	def __init__(self, Node_RWR, gene_set_file, Net_obj = None):
		if Net_obj is None:
			net_file_l = []
			net_file_l.append(data_dir + 'network/human/string_integrated.txt')
			Net_obj = BioNetwork(net_file_l)
		self.Node_RWR = Node_RWR
		network = Net_obj.sparse_network.toarray()
		i2g = Net_obj.i2g
		g2i = Net_obj.g2i
		self.nnode = len(i2g)
		self.path2gene = collections.defaultdict(dict)
		fin = open(gene_set_file)
		self.p2i = {}
		for i,line in enumerate(fin):
			f,g = line.strip().split('\t')
			if g.upper() not in g2i or f.startswith('GO_'):
				continue
			if f not in self.p2i:
				self.p2i[f] = len(self.p2i)
			self.path2gene[f][g2i[g.upper()]] = 1
		fin.close()
		self.i2p = {}
		for p in self.p2i:
			i = self.p2i[p]
			self.i2p[i] = p
		self.npath = len(self.path2gene)


	def RunDiffusion(self, p_train = 0.8,all_gene_cv=True,random_state=0):
		np.random.seed(random_state)
		test_ind_p = {}
		if all_gene_cv:
			train_ind, test_ind = train_test_split(range(self.nnode), test_size=1.-p_train)
			for i in range(self.npath):
				test_ind_p[i] = test_ind
		else:
			for i in range(self.npath):
				p = self.i2p[i]
				ng = int(len(self.path2gene[p]) * (1. - p_train) )
				test_ind_p[i] = np.random.choice(self.path2gene[p].keys(), ng, replace=False)
			train_ind = np.array(range(self.nnode))
			test_ind = np.array([])

		Path_RWR = np.zeros((self.npath, self.nnode))
		Path_mat_train = np.zeros((self.npath, self.nnode))
		Path_mat_test = np.zeros((self.npath, self.nnode))
		for f in self.path2gene:
			ct = 0
			for g in self.path2gene[f]:
				if g in test_ind_p[self.p2i[f]]:
					Path_mat_test[self.p2i[f],g] = 1
					continue
				Path_mat_train[self.p2i[f], g] = 1
				Path_RWR[self.p2i[f],:] += self.Node_RWR[g,:]
				ct += 1
			for g in test_ind_p[self.p2i[f]]:
				Path_mat_test[self.p2i[f],g] = 1
			if ct == 0:
				Path_RWR[self.p2i[f],:] = np.repeat(1./self.nnode,self.nnode)
			else:
				Path_RWR[self.p2i[f],:] /= ct
			i = self.p2i[f]
		return Path_RWR, Path_mat_train, Path_mat_test, train_ind, test_ind

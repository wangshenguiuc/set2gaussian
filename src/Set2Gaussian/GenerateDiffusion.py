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
from FunctionAnnotation import FunctionAnnotation
import scipy.spatial as sp
from sklearn.decomposition import SparsePCA
import time
from sklearn.cross_validation import train_test_split

class GenerateDiffusion:
	def __init__(self, Node_RWR,Net_obj = None, dataset='GO'):
		if Net_obj is None:
			net_file_l = []
			net_file_l.append(data_dir + 'network/human/string_integrated.txt')
			Net_obj = BioNetwork(net_file_l)
		self.dataset = dataset
		self.Node_RWR = Node_RWR
		network = Net_obj.sparse_network.toarray()
		i2g = Net_obj.i2g
		g2i = Net_obj.g2i
		self.nnode = len(i2g)


		if False and self.dataset == 'GO':
			GO_file_l = [data_dir + 'function_annotation/GO.network']
			GO_obj = BioNetwork(GO_file_l,reverse=True)
			GO_net = GO_obj.network_d[GO_file_l[0]]
			GO_rev_obj = BioNetwork(GO_file_l,reverse=False)
			GO_net_rev = GO_rev_obj.network_d[GO_file_l[0]]
			fin = open(data_dir+'function_annotation/GO2name.txt')
			GO2name  ={}
			name2GO  ={}
			for line in fin:
				w  = line.strip().split('\t')
				if len(w) < 2:
					continue
				GO2name[w[0]] = w[1]
				name2GO[w[1]] = w[0]
			fin.close()

			Func_obj = FunctionAnnotation(data_dir + 'function_annotation/gene_association.goa_human', GO_net)
			self.p2i = {}
			self.path2gene = collections.defaultdict(dict)
			for f in Func_obj.f2g:
				for g in Func_obj.f2g[f]:
					if g.upper() not in g2i:
						continue
					if f not in self.p2i:
						self.p2i[f] = len(self.p2i)
					self.path2gene[f][g2i[g.upper()]] = 1
		elif self.dataset=='gdsc' or self.dataset=='ctrp':
			self.path2gene,self.p2i,self.path2label = self.ReadDrugData(self.dataset,g2i)
		else:
			file = self.dataset + '_gene_set.txt'
			self.path2gene = collections.defaultdict(dict)
			fin = open(data_dir+'pathway/' + file)
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
		#print 'n gene set',self.dataset,self.npath

	def ReadDrugData(self,dataset,g2i,filter_ez_case=False,drug_cor_cutoff=0.2):
		if self.dataset == 'gdsc':
			fin = open('data/NLP_Dictionary/gdsctop_genes_exp_hgnc.txt')
			path2gene = {}
			p2i = {}
			i2p = {}
			nd = 0
			for line in fin:
				w = line.upper().strip().split('\t')
				cor = float(w[2])
				if abs(cor) < drug_cor_cutoff:
					continue
				d = w[1]
				if w[0] not in g2i:
					continue
				if d not in path2gene:
					path2gene[d] = {}
					p2i[d] = nd
					i2p[nd] = d
					nd += 1
				path2gene[d][g2i[w[0]]] = 1
			fin.close()
			#print len(path2gene)
			fin = open('data/drug/gdsc/drug_target_mapped.txt')
			path2label = {}
			for line in fin:
				w = line.upper().strip().split('\t')
				if len(w)==1:
					continue
				if w[0] not in p2i:
					continue

				gset = w[1].split(';')
				for i in gset:
					if i in g2i:
						if w[0] not in path2label:
							path2label[w[0]] = set()
						path2label[w[0]].add(g2i[i])
			fin.close()
		elif self.dataset == 'ctrp':
			fin = open('data/drug/ctrp/drug_map.txt')
			d2dname = {}
			for line in fin:
				w = line.upper().strip().split('\t')
				d2dname[w[0].replace('-','')] = w[2]
			fin.close()
			fin = open('data/NLP_Dictionary/ccletop_genes_exp_hgnc.txt')
			path2gene = {}
			p2i = {}
			i2p = {}
			nd = 0
			for line in fin:
				w = line.upper().strip().split('\t')
				cor = float(w[2])
				if abs(cor) < drug_cor_cutoff:
					continue
				d = d2dname[w[1]]
				if w[0] not in g2i:
					continue
				if d not in path2gene:
					path2gene[d] = {}
					p2i[d] = nd
					i2p[nd] = d
					nd += 1
				path2gene[d][g2i[w[0]]] = 1
			fin.close()
			fin = open('data/drug/ctrp/drug_target.txt')
			path2label = {}
			for line in fin:
				w = line.upper().strip().split('\t')
				if len(w)==1:
					continue
				if w[0] not in p2i:
					continue
				path2label[w[0]] = set()
				gset = w[1].split(';')
				for i in gset:
					if i in g2i:
						path2label[w[0]].add(g2i[i])
			fin.close()
		else:
			sys.exit('wrong dataset name')
		new_path2gene = {}
		new_path2label = {}
		new_p2i = {}
		if filter_ez_case:
			for p in path2label:
				find = False
				for lab in path2label[p]:
					if lab not in path2gene[p]:
						find =True
				if find:
					new_path2label[p] = path2label[p]
					new_path2gene[p] = path2gene[p]
					new_p2i[p] = len(new_p2i)
			print 'remove',len(path2label)-len(new_path2label),'npath',len(path2label),len(new_path2label)
		else:
			for p in path2label:
				new_path2label[p] = path2label[p]
				new_path2gene[p] = path2gene[p]
				new_p2i[p] = len(new_p2i)
			print 'remove',len(path2label)-len(new_path2label),'npath',len(path2label),len(new_path2label)

		return new_path2gene,new_p2i,new_path2label


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
		if self.dataset=='gdsc' or self.dataset=='ctrp':
			for i in range(self.npath):
				p = self.i2p[i]
				test_ind_p[i] = []
				for lab in self.path2label[p]:
					test_ind_p[i].append(lab)
				test_ind_p[i] = np.array(test_ind_p[i])
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
					if not (self.dataset=='gdsc' or self.dataset=='ctrp'):
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

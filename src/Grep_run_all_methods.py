import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import collections
import operator
import sys
import random
import gc
import os
from PreProcess_data import *

#Step 0: change gene_set_file and network_file to your files. The only parameters worth tunning are para_dict['nhidden'] (e.g., 1-5) and para_dict['node_emb_dim'] (e.g., 200-1000).
gene_set_file = '../data/node_set.txt'
network_file = '../data/network.txt' # file format: geneA\tgeneB\tconfidence\n
output_emb_file = 'output_embed'
para_dict = {}
para_dict['max_iter'] = 50
para_dict['node_emb_restart_prob'] = 0.8
para_dict['early_stopping'] = 20
para_dict['gene_loss_lambda'] = 100
para_dict['p_train'] = 1.
para_dict['lr'] = .0001
para_dict['nhidden'] = 3#2
para_dict['node_emb_dim'] = 500#3
para_dict['method'] = 'Set2Gaussian'
para_dict['optimize_path_mean'] = False
para_dict['dataset_name'] = 'nci'
para_dict['optimize_diag_path'] = 1

#First step: use network (e.g., PPI network) to calculate node embedding and node diffusion states.
Net_obj, Node_RWR, node_emb, node_context = read_node_embedding(para_dict['node_emb_dim'], network_file)

#Second step: read gene sets and calculate gene set diffusion states
Path_RWR, log_Path_RWR, log_node_RWR, train_ind, test_ind, Path_mat_train, _ = create_matrix(Node_RWR, Net_obj, para_dict['p_train'], gene_set_file)

#Third step: run gaussian embedding. Path_mu (2d array) is the mean embedding of each gene set. Path_cov (dictionary) is the covariance matrix of each gene set.
path_mu, path_cov, Grep_node_emb, p2g = run_embedding_method(para_dict['method'], log_Path_RWR, log_node_RWR, Path_RWR, node_emb, node_context,train_ind,test_ind,Path_mat_train,para_dict)

#Fourth step: save everything to the file.
save_mbedding(p2g, path_mu, path_cov, Grep_node_emb, output_file=output_emb_file)

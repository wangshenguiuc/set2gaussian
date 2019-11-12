import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import collections
import operator
import pickle
import sys
import random
import gc
import os
from PreProcess_data import *
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/'
sys.path.append(repo_dir)
os.chdir(repo_dir)


#200_200_200_0.5_0.001_0.9_0.5
#500_100_200_0.5_0.001_0.9_0.5_gdsc_drug_1_

if len(sys.argv) <= 2:
	DCA_dim = 2
	nhidden = 2
	max_iter = 50
	lr = 0.0001
	p_train = 0.5
	gene_loss_lambda = 100
	dataset = 'Reactome'
	early_stopping = 20
else:
	dataset = str(sys.argv[1])
	DCA_dim = int(sys.argv[2])
	nhidden = int(sys.argv[3])
	lr = float(sys.argv[4])
	p_train = float(sys.argv[5])
	gene_loss_lambda = float(sys.argv[6])
	max_iter = int(sys.argv[7])
	early_stopping = int(sys.argv[8])

optimize_diag_path = 1
optimize_path_mean = False
method = 'Grep'
DCA_rst = 0.8
para_dict = {}
para_dict['max_iter'] = max_iter
para_dict['DCA_rst'] = DCA_rst
para_dict['early_stopping'] = early_stopping
para_dict['gene_loss_lambda'] = gene_loss_lambda
para_dict['p_train'] = p_train
para_dict['lr'] = lr
para_dict['nhidden'] = nhidden
para_dict['DCA_dim'] = DCA_dim
para_dict['method'] = method
para_dict['optimize_path_mean'] = optimize_path_mean
if not isinstance(dataset, list):
	para_dict['dataset_name'] = dataset
else:
	para_dict['dataset_name'] = 'threedataset'
para_dict['optimize_diag_path'] = optimize_diag_path

print dataset, DCA_dim, nhidden, lr, p_train, gene_loss_lambda, max_iter, early_stopping

Net_obj, Node_RWR, node_emb, node_context = read_node_embedding(DCA_dim)

Path_RWR, log_Path_RWR, log_node_RWR, train_ind, test_ind, Path_mat_train, Path_mat_test = create_matrix(Node_RWR, Net_obj, p_train, dataset_l = dataset)

path_mu, path_cov, Grep_node_emb, p2g = run_embedding_method(method,log_Path_RWR, log_node_RWR, Path_RWR, node_emb, node_context,train_ind,test_ind,Path_mat_train,para_dict)

save_mbedding( p2g, path_mu, path_cov, Grep_node_emb, output_file='result/PathwayEmb/RecoverPath/MatrixOutput/'+str(optimize_diag_path)+'_'+str(DCA_rst)+'_'+para_dict['dataset_name']+'_'+str(DCA_dim) + '_' + str(nhidden) +  '_' +str(lr) +  '_' +str(p_train) +  '_' +str(gene_loss_lambda) + '_' + str(max_iter) +'_'+str(early_stopping))

if method!='Grep':
	metric = 'cosine'
	flog_name = 'result/PathwayEmb/RecoverPath/baseline_'+dataset + '_'+str(DCA_dim) +'_'+metric+'_'+str(DCA_rst) +'.txt'
else:
	flog_name = 'result/PathwayEmb/RecoverPath/'+str(optimize_diag_path)+'_'+str(DCA_rst)+'_threedataset_'+str(DCA_dim) + '_' + str(nhidden) +  '_' +str(lr) +  '_' +str(p_train) +  '_' +str(gene_loss_lambda) + '_' + str(max_iter) +'_'+str(early_stopping)+'.txt'
evaluate_embedding(flog_name, p2g, Path_mat_test, Path_mat_train)




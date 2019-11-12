import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .utils import *
#from tensorflow_probability import distributions as tfd
import sys
from scipy.optimize import minimize
from scipy import stats
import GPUtil
import time
import psutil
import FunctionPred as FP


class Graph2Gauss:
	def __init__(self, log_path_RWR, log_node_RWR, path_RWR, path_mat, node_emb, node_context, L=50, batch_size = 100,evalute_every_iter=False,path2gene=[],auc_d=[], p2i=[], path2label=[],optimize_gene_vec=False, lr = 1e-2, K=1, n_hidden=[50],use_piecewise_loss=False,gene_loss_lambda=1.,
				 max_iter=2000, tolerance=1000, scale=False, seed=0, verbose=True,eval_obj={}):
		"""
		Parameters
		----------
		A : scipy.sparse.spmatrix
			Sparse unweighted adjacency matrix
		X : scipy.sparse.spmatrix
			Sparse attribute matirx
		L : int
			Dimensionality of the node embeddings
		K : int
			Maximum distance to consider
		p_val : float
			Percent of edges in the validation set, 0 <= p_val < 1
		p_test : float
			Percent of edges in the test set, 0 <= p_test < 1
		p_nodes : float
			Percent of nodes to hide (inductive learning), 0 <= p_nodes < 1
		n_hidden : list(int)
			A list specifying the size of each hidden layer, default n_hidden=[512]
		max_iter :  int
			Maximum number of epoch for which to run gradient descent
		tolerance : int
			Used for early stopping. Number of epoch to wait for the score to improve on the validation set
		scale : bool
			Whether to apply the up-scaling terms.
		seed : int
			Random seed used to split the edges into train-val-test set
		verbose : bool
			Verbosity.
		"""

		tf.reset_default_graph()
		tf.set_random_seed(seed)
		np.random.seed(seed)
		self.evalute_every_iter = evalute_every_iter
		self.auc_d = auc_d
		self.lr = lr
		self.path2gene = path2gene
		self.optimize_gene_vec = optimize_gene_vec
		self.gene_loss_lambda = gene_loss_lambda
		#self.path_RWR = path_RWR.astype(np.float32)
		#self.path_mat = path_mat.astype(np.float32)
		np.random.seed(seed)
		self.nselect_path = np.shape(log_path_RWR)[0]
		self.p2i = p2i
		self.path2label = path2label
		self.log_path_RWR = log_path_RWR.astype(np.float32)
		self.log_node_RWR = log_node_RWR.astype(np.float32)
		self.path_RWR = tf.convert_to_tensor(path_RWR.astype(np.float32))
		self.path_mat = path_mat.astype(np.float32)
		self.path_weight = np.exp(1. / np.sum(self.path_mat,axis=1))

		self.path_mat = tf.convert_to_tensor(self.path_mat.astype(np.float32))

		print np.shape(self.path_weight)
		self.path_weight = tf.convert_to_tensor(self.path_weight.astype(np.float32))
		self.use_piecewise_loss = use_piecewise_loss
		self.npath, self.nnode = self.log_path_RWR.shape
		self.log_node_RWR = tf.convert_to_tensor(self.log_node_RWR)
		node_emb = node_emb.astype(np.float32)
		node_context = node_context.astype(np.float32)
		self.batch_size = batch_size
		self.D = node_emb.shape[1]
		self.L = L
		self.reg_mu_init = 0
		self.max_iter = max_iter
		self.tolerance = tolerance
		self.scale = scale
		self.verbose = verbose
		self.eval_obj = eval_obj

		if n_hidden is None:
			n_hidden = [512]
		self.n_hidden = n_hidden


		self.__build( node_emb, node_context)
		sys.stdout.flush()
		#self.__dataset_generator(hops, scale_terms)
		self.__build_loss()
		sys.stdout.flush()
		# setup the validation set for easy evaluation

	def __build(self, node_mu, node_context):
		w_init = tf.contrib.layers.xavier_initializer
		if self.optimize_gene_vec:
			#self.node_mu = tf.get_variable(name='node_mu', dtype=tf.float32, shape=[self.nnode, self.L])
			self.node_mu = tf.get_variable(name='node_mu', initializer=node_mu)
			#self.node_context = tf.get_variable(name='node_context', initializer =node_context)
			self.node_context = tf.get_variable(name='node_context', initializer= node_context)
			#self.node_context = tf.get_variable(name='node_context', dtype=tf.float32, shape=[self.nnode, self.L])
		else:
			self.node_mu = tf.convert_to_tensor(node_mu)
			self.node_context = tf.convert_to_tensor(node_context)


	def __build_loss(self):
		self.slice_st = tf.placeholder(tf.int32)
		self.slice_ed = tf.placeholder(tf.int32)

		path_mu = tf.matmul(self.path_mat[self.slice_st:self.slice_ed,:], self.node_mu)
		self.pathway_loss = 0.
		path_mu_batch = path_mu
		path_mu_batch = tf.expand_dims(path_mu_batch,1)
		path_bar_batch_all_node = tf.tile(path_mu_batch,[1,self.nnode,1])

		node_mu = tf.expand_dims(self.node_mu,0)
		node_bar_batch_all_node = tf.tile(node_mu,[self.slice_ed - self.slice_st,1,1])

		node_path_diff = path_bar_batch_all_node - node_bar_batch_all_node #p*n*d
		#node_path_diff = tf.expand_dims(pgd,3)#p*n*d to p*n*d
		node_avg = tf.multiply(node_path_diff, node_path_diff) #p*n*d
		wt = tf.convert_to_tensor(self.path_mat[self.slice_st:self.slice_ed,:] ) #p*n
		wt = tf.expand_dims(wt,2)
		wt_mat = tf.tile(wt,[1,1,self.L]) #p*n*d
		wt_node_avg = tf.multiply(node_avg, wt_mat) #p*n*d
		path_wt = self.path_weight[self.slice_st:self.slice_ed]#p
		path_wt = tf.expand_dims(path_wt,1)#p*1
		path_wt = tf.tile(path_wt,[1,self.L])
		self.cov = tf.reduce_sum(wt_node_avg, 1, keep_dims=False) #p*d

		wt_node_avg = tf.multiply(self.cov, path_wt) #p*n*d
		self.pathway_loss +=  tf.reduce_sum(wt_node_avg)

		#g2g_diff = tf.matmul(self.node_mu, tf.transpose(self.node_context))
		#if self.optimize_gene_vec:
		#	self.gene_loss = self.gene_loss_lambda * tf.losses.mean_squared_error(g2g_diff, self.log_node_RWR)
		#else:
		#	self.gene_loss = 0.

		self.loss = self.pathway_loss


	def __build_loss_unused(self):

		path_mu = tf.matmul(self.path_RWR, self.node_mu)
		self.pathway_loss = 0.
		for p in range(self.npath):
			path_mean = path_mu[p,:]
			path_mean = tf.expand_dims(path_mean,0)
			#print path_mean.get_shape().as_list()
			node_path_diff = self.node_mu - tf.tile(path_mean,[self.nnode,1]) #n*d
			node_path_diff = tf.expand_dims(node_path_diff,2)#n*d to n*d*1
			node_path_diff_trans = tf.transpose(node_path_diff, perm=[0,2,1]) # n*1*d
			node_avg = tf.matmul(node_path_diff, node_path_diff_trans) #n*d*d
			wt = self.path_RWR[p,:]
			wt = tf.expand_dims(wt,0)
			wt = tf.expand_dims(wt,0)
			wt = tf.transpose(wt, perm=[2,0,1])
			#print wt.get_shape().as_list()
			wt_mat = tf.tile(wt,[1,self.L,self.L]) #n*d*d
			wt_node_avg = tf.multiply(node_avg, wt_mat) #n*d*d
			cov = tf.reduce_sum(wt_node_avg, 0, keep_dims=False) #d*d
			diag = tf.diag_part(cov) #d*1
			self.pathway_loss +=  tf.reduce_sum(cov)

		g2g_diff = tf.matmul(self.node_mu, tf.transpose(self.node_context))
		if self.optimize_gene_vec:
			self.gene_loss = self.gene_loss_lambda * tf.losses.mean_squared_error(g2g_diff, self.log_node_RWR)
		else:
			self.gene_loss = 0.

		self.loss = self.gene_loss


	def test_loss(self,sess):
		cov_inv_int = tf.convert_to_tensor(np.array([[0.22375,0],[0,0.22375]]).astype(np.float32))
		cov_inv_int = tf.gather(self.cov_inv,0)
		cov_inv_k = cov_inv_int
		path_mu_batch = tf.gather(self.path_mu,0)
		path_mu_batch = tf.expand_dims(path_mu_batch,1)
		path_bar_batch_all_node = tf.tile(path_mu_batch,[1,self.nnode,1])

		node_mu = tf.expand_dims(self.node_mu,0)
		node_bar_batch_all_node = tf.tile(node_mu,[self.batch_size,1,1])

		pgd = self.node_mu
		print 'PATH here',sess.run(self.node_mu)
		print 'NODE here',sess.run(self.node_mu)
		pdd = cov_inv_k
		print 'pdd here',sess.run(cov_inv_k)
		pgd_pdd =  tf.matmul(pgd, pdd)
		print 'pgd_pdd here',sess.run(pgd_pdd)
		print 'pgd here',sess.run(pgd)
		tmp1 = tf.multiply( pgd_pdd, pgd)
		print 'tmp1 here',sess.run(tmp1)
		tmp_p2g  = tf.reduce_sum( tmp1, 1, keep_dims=False )
		print 'here',sess.run(tmp_p2g)
		print 'truth',self.log_path_RWR
		#return sess.run(tmp_p2g)


	def calculate_set_loss(self):
		p2g_prob_discrete = tf.matmul(self.path_RWR, self.node_mu)
		set_loss_discrete = tf.losses.mean_squared_error(self.path_mu, p2g_prob_discrete)
		p2g_prob_continue = tf.matmul(self.path_mat, self.node_mu)
		set_loss_continue = tf.losses.mean_squared_error(self.path_mu, p2g_prob_continue)
		return set_loss_discrete,set_loss_continue

	def calculate_var_mean(self,sess):
		path_cov = {}
		path_mu = {}
		path_mu_exp = {}
		path_cov_inv = {}
		path_cov_w_batch = self.path_cov_w
		path_cov_x_batch = self.path_cov_x

		path_cov_x_trans = tf.transpose(path_cov_x_batch, perm=[0,2,1])
		C_k = tf.matmul(path_cov_w_batch, path_cov_x_trans)
		C_k_trans = tf.transpose(C_k, perm=[0,2,1])
		cov_inv_k =  tf.matmul(C_k, C_k_trans)
		#cov_inv_k = self.cov_inv
		#path_cov_x_trans = tf.transpose(self.path_cov_x, perm=[0,2,1])
		#C_k = tf.matmul(self.path_cov_w, path_cov_x_trans)
		#C_k_trans = tf.transpose(C_k, perm=[0,2,1])
		#cov_inv_k =  tf.matmul(C_k, C_k_trans)
		#sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
		#sess.run(tf.global_variables_initializer())
		for pi in range(self.npath):
			cov_inv_k_tmp = tf.gather(cov_inv_k,pi)
			cov = tf.matrix_inverse(cov_inv_k_tmp)
			path_cov_inv[pi] = sess.run(cov_inv_k_tmp)
			path_cov[pi] = sess.run(cov)
			path_mu_exp[pi] = sess.run(tf.gather(tf.matmul(self.path_RWR, self.node_mu), pi))
			path_mu[pi] = sess.run(tf.gather(self.path_mu, pi))
		node_emb = sess.run(self.node_mu)
		return path_cov, path_mu, path_cov_inv, path_mu_exp, node_emb

	def __save_vars(self, sess):
		"""
		Saves all the trainable variables in memory. Used for early stopping.

		Parameters
		----------
		sess : tf.Session
			Tensorflow session used for training
		"""
		self.saved_vars = {var.name: (var, sess.run(var)) for var in tf.trainable_variables()}

	def __restore_vars(self, sess):
		"""
		Restores all the trainable variables from memory. Used for early stopping.
		Parameters
		----------
		sess : tf.Session
			Tensorflow session used for training
		"""
		for name in self.saved_vars:
				sess.run(tf.assign(self.saved_vars[name][0], self.saved_vars[name][1]))


	def train(self, gpu_list='0'):
		"""
		Trains the model.

		Parameters
		----------
		gpu_list : string
			A list of available GPU devices.

		Returns
		-------
		sess : tf.Session
			Tensorflow session that can be used to obtain the trained embeddings

		"""
		early_stopping_score_max = -1.0
		tolerance = self.tolerance
		max_auc = -1
		max_epoch = -1
		max_auc_sim = -1
		max_auc_sim_epoch = -1
		max_std = -1
		best_pv_rwr = 1
		best_pv_dca = 1
		train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

		sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
		sess.run(tf.global_variables_initializer())

		#print np.sum([
        #    np.prod(v.get_shape().as_list())
        #    for v in tf.trainable_variables()
        #])
		self.best_para = {}
		batch_size = self.batch_size
		set_loss_dis_sc = 0
		set_loss_cont_sc = 0
		for epoch in range(self.max_iter):
			start_time = time.time()
			#auc_l = self.eval_obj.evaluate(sess.run(self.node_mu),low_b=[1],up_b=[300])
			#gene_loss = sess.run(self.gene_loss)
			#print self.gene_loss_lambda,gene_loss,np.mean(auc_l)
			#loss_all, _ = sess.run([self.loss, train_op])
			loss_all = 0.

			for bi in range(0,self.npath,batch_size):
				st = bi
				ed = bi + batch_size
				if ed > self.nselect_path:
					break

				ed = min(ed,self.nselect_path)
				#print st,ed
				loss, _ = sess.run([self.loss, train_op],
				feed_dict={self.slice_st:st,
				self.slice_ed:ed})
				loss_all += loss

			if epoch % 1 == 0:

				auc_l = self.eval_obj.evaluate(sess.run(self.node_mu),GO_emb_vec=sess.run(self.path_mu))
				print np.mean(auc_l)
				#gene_loss = sess.run(self.gene_loss)
				#print gene_loss,sess.run(self.cov,feed_dict={self.slice_st:1,self.slice_ed:2})
				#print self.gene_loss_lambda,epoch, np.mean(auc_l),loss_all,gene_loss
				print np.nanmean(auc_l)
				sys.stdout.flush()

			'''
			#print path_cov


			if epoch%1000==0 and epoch>0:
				#np_loss = self.test_loss(sess)
				p2g = sess.run(self.p2g,feed_dict={self.slice_st:st,
					self.slice_ed:ed,
					self.Y_target:self.log_path_RWR[st:ed,:]})
				print p2g
				print self.log_path_RWR
				path_cov, path_mu, path_cov_inv, path_mu_exp, node_emb = self.calculate_var_mean(sess)
				for k in range(1):
					print k, path_cov[k],  path_cov_inv[k], path_mu[k], path_mu_exp[k]
				print('epoch: {:3d}, loss: {:.10f}'.format(epoch, loss_all))
			#set_loss_dis, set_loss_cont = self.calculate_set_loss()
			#set_loss_dis_sc = sess.run(set_loss_dis)
			#set_loss_cont_sc = sess.run(set_loss_cont)
			#process = psutil.Process(os.getpid())
			#print 'CPU',process.memory_info().rss/1024./1024./1024.,'GB'
			#GPUtil.showUtilization()
			'''



		#self.__restore_vars(sess)
		#path_cov, path_mu, Path_our =self.get_guassian_distr_para(sess)
		#p2g = sess.run(self.p2g,feed_dict={self.slice_st:0,
		#			self.slice_ed:self.npath,
		#			self.Y_target:self.log_path_RWR[0:self.npath,:]})
		#path_cov, path_mu, path_cov_inv, path_mu_exp, node_emb = self.calculate_var_mean(sess)
		#return p2g,path_mu,path_cov,node_emb

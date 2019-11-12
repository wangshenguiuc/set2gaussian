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


class Graph2Gauss:
	def __init__(self, log_Path_RWR, log_node_RWR, Path_RWR, Path_mat_train, Path_mat_test, node_emb, node_context, L=50, path_batch_size = 100,node_batch_size = 100,evalute_every_iter=False,path2gene=[],auc_d=[], p2i=[], path2label=[],optimize_gene_vec=False,optimize_path_mean=True,optimize_diag_path=0,lr = 1e-2, K=1, n_hidden=[50],use_piecewise_loss=False,gene_loss_lambda=1.,
				 max_iter=2000, tolerance=1000, scale=False, seed=0, verbose=True,eval_obj={}, train_ind=[],
			 	test_ind = [], flog='',EVALUATE=True):
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
		self.EVALUATE = EVALUATE
		self.lr = lr
		self.flog = flog
		if self.flog == '':
			self.flog = open('test','w')
		self.path2gene = path2gene
		self.optimize_gene_vec = optimize_gene_vec
		self.optimize_path_mean = optimize_path_mean
		self.optimize_diag_path = optimize_diag_path
		self.gene_loss_lambda = gene_loss_lambda
		self.path_batch_size = path_batch_size
		self.node_batch_size = node_batch_size
		#self.Path_RWR = Path_RWR.astype(np.float32)
		#self.Path_mat_train = Path_mat_train.astype(np.float32)
		np.random.seed(seed)
		self.p2i = p2i
		self.path2label = path2label
		self.log_Path_RWR = log_Path_RWR.astype(np.float32)
		self.log_node_RWR = log_node_RWR.astype(np.float32)
		self.Path_RWR = Path_RWR.astype(np.float32)
		self.Path_mat_train = Path_mat_train.astype(np.float32)
		self.Path_mat_test = Path_mat_test.astype(np.float32)
		self.use_piecewise_loss = use_piecewise_loss
		self.npath, self.nnode = self.log_Path_RWR.shape
		#self.log_Path_RWR = tf.convert_to_tensor(self.log_Path_RWR)
		self.log_node_RWR = tf.convert_to_tensor(self.log_node_RWR)
		node_emb = node_emb.astype(np.float32)
		self.node_emb = node_emb
		node_context = node_context.astype(np.float32)
		self.D = node_emb.shape[1]
		self.L = L
		self.reg_mu_init = 0
		self.max_iter = max_iter
		self.tolerance = tolerance
		self.scale = scale
		self.verbose = verbose
		self.eval_obj = eval_obj

		self.train_ind = train_ind
		self.test_ind = test_ind

		self.ntest = len(self.test_ind)
		self.ntrain = len(self.train_ind)

		if n_hidden is None:
			n_hidden = [512]
		self.n_hidden = n_hidden

		self.__build( node_emb, node_context)
		sys.stdout.flush()
		#self.__dataset_generator(hops, scale_terms)
		self.__build_loss()
		sys.stdout.flush()
		# setup the validation set for easy evaluation

	def __build(self, node_mu, node_context):#0:diag path cov, 1: full path cov, 2:one side optimize, 3:three way dot
		w_init = tf.contrib.layers.xavier_initializer
		if self.optimize_gene_vec:
			#self.node_mu = tf.get_variable(name='node_mu', dtype=tf.float32, shape=[self.nnode, self.L],initializer= w_init())
			self.node_mu = tf.get_variable(name='node_mu', initializer=node_mu)
			#self.node_context = tf.get_variable(name='node_context', dtype=tf.float32, shape=[self.nnode, self.L],initializer= w_init())
			self.node_context = tf.get_variable(name='node_context', initializer= node_context)
		else:
			self.node_mu = tf.convert_to_tensor(node_mu)
			self.node_context = tf.convert_to_tensor(node_context)
		if self.optimize_diag_path==2:
			self.path_cov = tf.ones([self.npath, self.L], tf.float32)
		elif self.optimize_diag_path==1:
			self.path_cov_w = tf.get_variable(name='Path_sigma_w', dtype=tf.float32, shape=[self.npath, self.L,  self.n_hidden[0]],initializer= w_init())
			self.path_cov_x = tf.get_variable(name='Path_sigma_x', dtype=tf.float32, shape=[self.npath, self.L , self.n_hidden[0]],initializer= w_init())
		elif self.optimize_diag_path==0:
			self.path_cov = tf.get_variable(name='Path_sigma_w', dtype=tf.float32, shape=[self.npath, self.L],initializer= w_init())
		elif self.optimize_diag_path==3:
			self.path_context = tf.get_variable(name='Path_context', dtype=tf.float32, shape=[self.npath, self.L],initializer= w_init())

		if self.optimize_path_mean:
			Path_emb = np.dot( self.Path_RWR,node_mu)
			self.path_mu = tf.get_variable(name='Path_mu',initializer= Path_emb)
		else:
			self.path_mu = tf.get_variable(name='Path_mu',dtype=tf.float32, shape=[self.npath, self.L],initializer= w_init())


	def __build_loss(self):
		self.path_slice_st = tf.placeholder(tf.int32)
		self.path_slice_ed = tf.placeholder(tf.int32)
		self.gene_slice_st = tf.placeholder(tf.int32)
		self.gene_slice_ed = tf.placeholder(tf.int32)
		self.node_ind_old = tf.placeholder(shape=[None], dtype=tf.int32)
		self.node_ind = tf.convert_to_tensor(self.node_ind_old)
		node_ind_L = tf.shape(self.node_ind)[0]
		#node_ind = self.train_ind[self.gene_slice_st:self.gene_slice_ed]
		self.Y_target = tf.placeholder(shape=[None, None], dtype=tf.float32)
		if self.optimize_diag_path==2:
			path_context = self.path_context[self.path_slice_st:self.slice_ed,:]
			path_mu = self.path_mu[self.path_slice_st:self.slice_ed,:]
			pmc = tf.multiply( path_mu, path_context )
			p2p = tf.reduce_sum(pmc , 1, keep_dims=False)

			p2p = tf.expand_dims(p2p,1)
			p2p = tf.tile(p2p,[1,self.nnode])

			p2g = tf.matmul(path_context, tf.transpose(self.node_mu))
			self.p2g = p2p - p2g
		elif self.optimize_diag_path==0:
			self.cov_inv_k = tf.diag(self.path_cov[self.path_slice_st:self.slice_ed,:])
			#self.cov_inv_k = self.cov_inv[self.slice_st:self.slice_ed,:,:]
			path_mu_batch = self.path_mu[self.path_slice_st:self.slice_ed,:]
			path_mu_batch = tf.expand_dims(path_mu_batch,1)
			path_bar_batch_all_node = tf.tile(path_mu_batch,[1,self.nnode,1])

			node_mu = tf.expand_dims(self.node_mu,0)
			node_bar_batch_all_node = tf.tile(node_mu,[self.slice_ed-self.path_slice_st,1,1])
			pgd = path_bar_batch_all_node - node_bar_batch_all_node
			pdd = self.cov_inv_k
			pgd_pdd = tf.matmul(pgd, pdd)
			self.p2g = tf.reduce_sum( tf.multiply( pgd, pgd_pdd ), 2, keep_dims=False )
		elif self.optimize_diag_path==1:
			path_cov_w_batch = self.path_cov_w[self.path_slice_st:self.path_slice_ed,:,:]
			path_cov_x_batch = self.path_cov_x[self.path_slice_st:self.path_slice_ed,:,:]
			path_cov_x_trans = tf.transpose(path_cov_x_batch, perm=[0,2,1])
			C_k = tf.matmul(path_cov_w_batch, path_cov_x_trans)
			C_k_trans = tf.transpose(C_k, perm=[0,2,1])
			self.cov_inv_k =  tf.matmul(C_k, C_k_trans)

			path_mu_batch = self.path_mu[self.path_slice_st:self.path_slice_ed,:]
			path_mu_batch = tf.expand_dims(path_mu_batch,1)
			path_bar_batch_all_node = tf.tile(path_mu_batch,[1,node_ind_L,1])
			#node_mu = self.node_mu[self.node_ind,:]
			node_mu = tf.gather(self.node_mu,self.node_ind)

			node_mu = tf.expand_dims(node_mu,0)
			node_bar_batch_all_node = tf.tile(node_mu,[self.path_slice_ed-self.path_slice_st,1,1])
			pgd = path_bar_batch_all_node - node_bar_batch_all_node
			pdd = self.cov_inv_k
			pgd_pdd = tf.matmul(pgd, pdd)
			self.p2g = tf.reduce_sum( tf.multiply( pgd, pgd_pdd ), 2, keep_dims=False )

		elif self.optimize_diag_path==3:
			path_context = self.path_context[self.path_slice_st:self.slice_ed,:]
			path_context = tf.multiply( path_context, path_context )
			path_mu = self.path_mu[self.path_slice_st:self.slice_ed,:]
			pmc = tf.multiply( path_mu, path_context )
			p2g = tf.matmul(pmc, tf.transpose(self.node_mu))
			self.p2g = p2p - p2g

		if self.use_piecewise_loss:
			self.pathway_loss = tf.losses.mean_pairwise_squared_error(self.p2g , self.Y_target)
		else:
			self.pathway_loss = tf.losses.mean_squared_error(self.p2g, self.Y_target)

		if self.optimize_gene_vec:
			node_mu = tf.gather(self.node_mu,self.node_ind)
			g2g_diff = tf.matmul(node_mu, tf.transpose(self.node_context))
			log_node_RWR_batch = tf.gather(self.log_node_RWR, self.node_ind)
			self.gene_loss = self.gene_loss_lambda * tf.losses.mean_squared_error(g2g_diff, log_node_RWR_batch)
		else:
			self.gene_loss = 0.

		self.loss = self.pathway_loss + self.gene_loss
		#self.loss = self.gene_loss


	def test_loss(self,sess):
		cov_inv_int = tf.convert_to_tensor(np.array([[0.22375,0],[0,0.22375]]).astype(np.float32))
		cov_inv_int = tf.gather(self.cov_inv,0)
		self.cov_inv_k = cov_inv_int
		path_mu_batch = tf.gather(self.path_mu,0)
		path_mu_batch = tf.expand_dims(path_mu_batch,1)
		path_bar_batch_all_node = tf.tile(path_mu_batch,[1,self.nnode,1])

		node_mu = tf.expand_dims(self.node_mu,0)
		node_bar_batch_all_node = tf.tile(node_mu,[self.batch_size,1,1])

		pgd = self.node_mu
		print 'PATH here',sess.run(self.node_mu)
		print 'NODE here',sess.run(self.node_mu)
		pdd = self.cov_inv_k
		print 'pdd here',sess.run(self.cov_inv_k)
		pgd_pdd =  tf.matmul(pgd, pdd)
		print 'pgd_pdd here',sess.run(pgd_pdd)
		print 'pgd here',sess.run(pgd)
		tmp1 = tf.multiply( pgd_pdd, pgd)
		print 'tmp1 here',sess.run(tmp1)
		tmp_p2g  = tf.reduce_sum( tmp1, 1, keep_dims=False )
		print 'here',sess.run(tmp_p2g)
		print 'truth',self.log_Path_RWR
		#return sess.run(tmp_p2g)


	def calculate_set_loss(self):
		p2g_prob_discrete = tf.matmul(self.Path_RWR, self.node_mu)
		set_loss_discrete = tf.losses.mean_squared_error(self.path_mu, p2g_prob_discrete)
		p2g_prob_continue = tf.matmul(self.Path_mat_train, self.node_mu)
		set_loss_continue = tf.losses.mean_squared_error(self.path_mu, p2g_prob_continue)
		return set_loss_discrete,set_loss_continue

	def calculate_var_mean(self,sess,k=3):
		path_cov = {}
		path_mu = {}
		path_mu_exp = {}
		path_cov_inv = {}
		path_cov_w_batch = self.path_cov_w[0:k,:]
		path_cov_x_batch = self.path_cov_x[0:k,:]

		path_cov_x_trans = tf.transpose(path_cov_x_batch, perm=[0,2,1])
		C_k = tf.matmul(path_cov_w_batch, path_cov_x_trans)
		C_k_trans = tf.transpose(C_k, perm=[0,2,1])
		self.cov_inv_k =  tf.matmul(C_k, C_k_trans)
		#self.cov_inv_k = self.cov_inv
		#path_cov_x_trans = tf.transpose(self.path_cov_x, perm=[0,2,1])
		#C_k = tf.matmul(self.path_cov_w, path_cov_x_trans)
		#C_k_trans = tf.transpose(C_k, perm=[0,2,1])
		#self.cov_inv_k =  tf.matmul(C_k, C_k_trans)
		#sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
		#sess.run(tf.global_variables_initializer())
		Path_RWR_c = tf.matmul(self.Path_RWR[0:k,:], self.node_mu)
		for pi in range(k):
			self.cov_inv_k_tmp = tf.gather(self.cov_inv_k,pi)
			cov = tf.matrix_inverse(self.cov_inv_k_tmp)
			path_cov_inv[pi] = sess.run(self.cov_inv_k_tmp)
			path_cov[pi] = sess.run(self.cov_inv_k_tmp)
			path_mu_exp[pi] = sess.run(tf.gather(Path_RWR_c,pi))
			path_mu[pi] = sess.run(tf.gather(self.path_mu,pi))
		#node_emb = sess.run(self.node_mu)
		node_emb = []
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
		for name in self.saved_vars:
				sess.run(tf.assign(self.saved_vars[name][0], self.saved_vars[name][1]))

	def evaluate(self):

		pathway_loss_all+= sess.run(self.pathway_loss,feed_dict={self.path_slice_st:pst,self.path_slice_ed:ped,self.gene_slice_st:gst,self.gene_slice_ed:ged,self.Y_target:self.log_Path_RWR[pst:ped,gst:ged]})

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
		final_path_cov = {}
		final_g2g_node_emb = {}
		final_p2g = {}
		final_path_mu = {}

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


		Path_emb = np.dot( self.Path_RWR, self.node_emb)
		Path_avg_emb = sp.distance.cdist(Path_emb, self.node_emb, 'cosine')

		self.best_para = {}
		set_loss_dis_sc = 0
		set_loss_cont_sc = 0
		st_auc = 0
		best_auc = 0
		for epoch in range(self.max_iter):
			start_time = time.time()
			if epoch % 1 == 0 and self.EVALUATE:
				if self.L != 2:
					if epoch == 0:
						auroc_d, base_auroc_l,prec_d,base_prec_l = evaluate_pathway_member(self.log_Path_RWR, self.Path_mat_test,self.Path_mat_train )
						print 'baseline auroc',auroc_d,prec_d
						self.flog.write('RWR')
						for d in auroc_d.keys():
							aup,l = auroc_d[d]
							prec, _ = prec_d[d]
							self.flog.write('\t'+str(aup)+'\t'+str(prec))
						self.flog.write('\n')
						auroc_d, auroc_l,prec_d,prec_l = evaluate_pathway_member(Path_avg_emb, self.Path_mat_test,self.Path_mat_train )
						print 'avg baseline auroc',auroc_d,prec_d
						self.flog.write('Avg')
						for d in auroc_d.keys():
							aup,l = auroc_d[d]
							prec, _ = prec_d[d]
							self.flog.write('\t'+str(aup)+'\t'+str(prec))
						self.flog.write('\n')

					auroc_l_all = {}
					prec_l_all = {}
					for pi in range(0,self.npath,self.path_batch_size):
						pst = pi
						ped = pi + self.path_batch_size
						ped = min(ped,self.npath)
						p2g = sess.run(self.p2g,feed_dict={self.path_slice_st:pst,self.path_slice_ed:ped,self.node_ind:range(self.nnode),
						self.Y_target:self.log_Path_RWR[pst:ped,np.array(range(self.nnode))]})
						auroc_d, auroc_l,prec_d,prec_l = evaluate_pathway_member(p2g, self.Path_mat_test[pst:ped,], self.Path_mat_train[pst:ped,])
						for b in auroc_l:
							if b not in auroc_l_all:
								auroc_l_all[b] = []
								prec_l_all[b] = []
							auroc_l_all[b].extend(auroc_l[b])
							prec_l_all[b].extend(prec_l[b])
					auroc_d = {}
					prec_d = {}
					for b in auroc_l_all:
						auroc_d[b] = np.nanmean(auroc_l_all[b])
						prec_d[b] = np.nanmean(prec_l_all[b])
					print 'our auroc',auroc_d,prec_d
					self.flog.write('our:'+str(epoch))
					for d in auroc_d.keys():
						aup = auroc_d[d]
						prec = prec_d[d]
						aup_pv = stats.ttest_ind(auroc_l_all[d],base_auroc_l[d])[1] / 2.
						prec_pv = stats.ttest_ind(prec_l_all[d],base_prec_l[d])[1] / 2.
						aup_pv1 = stats.ttest_rel(auroc_l_all[d],base_auroc_l[d])[1] / 2.
						prec_pv1 = stats.ttest_rel(prec_l_all[d],base_prec_l[d])[1] / 2.
						if np.mean(auroc_l_all[d]) < np.mean(base_auroc_l[d]):
							aup_pv = 1.
							aup_pv1 = 1.
						if np.mean(prec_l_all[d]) < np.mean(base_prec_l[d]):
							prec_pv = 1.
							prec_pv1 = 1.
						self.flog.write('\t'+str(aup)+'\t'+str(prec) + '\t' + str(aup_pv) + '\t'+str(aup_pv1) + '\t' + str(prec_pv)+ '\t' + str(prec_pv1))
					self.flog.write('\n')
					self.flog.flush()
					auc_l = []
					auc_l, pv_l = self.eval_obj.evaluate(sess.run(self.node_mu),GO_emb_vec=self.eval_obj.GO_emb_vec,test_ind=self.test_ind,train_ind=self.train_ind,low_b=[1],up_b=[300])#
					auc = np.mean(auc_l)
					if epoch==0:
						st_auc = auc
					else:
						if auc < st_auc - 0.01:
							pass
							#break
					if auc > best_auc:
						best_auc = auc



				test_pathway_loss = 0.
				test_gene_loss = 0
				#print len(self.test_ind)
				if len(self.test_ind) > 0:
					for pi in range(0,self.npath,self.path_batch_size):
						pst = pi
						ped = pi + self.path_batch_size
						ped = min(ped,self.npath)
						for bi in range(0,self.ntest,self.node_batch_size):
							gst = bi
							ged = bi + self.node_batch_size
							ged = min(ged,self.ntest)
							test_pathway_loss += sess.run(self.pathway_loss,feed_dict={self.path_slice_st:pst,self.path_slice_ed:ped,self.node_ind:self.test_ind[gst:ged],
							self.Y_target:self.log_Path_RWR[pst:ped,self.test_ind[gst:ged]]})
							#test_pathway_loss_tmp
							#print pst,ped,gst,ged,test_pathway_loss
					#print len(self.test_ind)
					test_gene_loss = 0
					if self.optimize_gene_vec:
						test_gene_loss = sess.run(self.gene_loss,feed_dict={self.path_slice_st:0,self.path_slice_ed:2,self.node_ind:self.test_ind,self.Y_target:self.log_Path_RWR[0:2,self.test_ind]})


				train_pathway_loss = 0.
				for pi in range(0,self.npath,self.path_batch_size):
					pst = pi
					ped = pi + self.path_batch_size
					ped = min(ped,self.npath)
					for bi in range(0,self.ntrain,self.node_batch_size):
						gst = bi
						ged = bi + self.node_batch_size
						ged = min(ged,self.ntrain)
						train_pathway_loss+= sess.run(self.pathway_loss,feed_dict={self.path_slice_st:pst,self.path_slice_ed:ped,self.node_ind:self.train_ind[gst:ged],
						self.Y_target:self.log_Path_RWR[pst:ped,self.train_ind[gst:ged]]})
				train_gene_loss = 0.
				if self.optimize_gene_vec:
					train_gene_loss = sess.run(self.gene_loss,feed_dict={self.path_slice_st:0,self.path_slice_ed:2,self.node_ind:self.train_ind,self.Y_target:self.log_Path_RWR[0:2,self.train_ind]})
				train_loss_all = train_gene_loss + train_pathway_loss
				if self.L == 2:
					auc = 0.5
					best_auc = 0.5
					best_auc = 0.5
				print('epoch: {:3d}, auc:{:.3f}, best:{:.3f}, st_auc:{:.3f},lambda:{:3e},train_all:{:.3e},path_train:{:.3f},path_test:{:.3f},gene_train:{:.3f},gene_test:{:.3f}'.format(epoch, auc,best_auc,st_auc,self.gene_loss_lambda,train_loss_all,train_pathway_loss,test_pathway_loss,train_gene_loss,test_gene_loss))
				if auc > st_auc:
					pass
					#print pv_l
					#print auc_l
					#print self.eval_obj.baseline_auc_number
				sys.stdout.flush()
				#process = psutil.Process(os.getpid())
				#print 'CPU',process.memory_info().rss/1024./1024./1024.,'GB'
				#GPUtil.showUtilization()

				'''
				p2g = sess.run(self.p2g,feed_dict={self.slice_st:0,
					self.slice_ed:2,
					self.Y_target:self.log_Path_RWR[0:2,:]})
				print p2g
				print self.log_Path_RWR[0:2,:]
				path_cov, path_mu, path_cov_inv, path_mu_exp, _ = self.calculate_var_mean(sess,k=3)
				for k in range(2):
					print k, path_cov[k],  path_cov_inv[k], path_mu[k], path_mu_exp[k]
				'''
			loss = 0.
			for pi in range(0,self.npath,self.path_batch_size):
				pst = pi
				ped = pi + self.path_batch_size
				ped = min(ped,self.npath)
				for bi in range(0,self.ntrain,self.node_batch_size):
					gst = bi
					ged = bi + self.node_batch_size
					ged = min(ged,self.ntrain)
					#print epoch, pi, bi
					#print pst, ped, gst, ged
					tmp_loss, _ = sess.run([self.loss, train_op],feed_dict={self.path_slice_st:pst,self.path_slice_ed:ped,self.node_ind:self.train_ind[gst:ged],self.Y_target:self.log_Path_RWR[pst:ped,self.train_ind[gst:ged]]})
					loss += tmp_loss
			print epoch,loss
			sys.stdout.flush()
			if self.L == 2:
				pst = 0
				ped = self.npath
				path_mu = sess.run(self.path_mu,feed_dict={self.path_slice_st:pst,self.path_slice_ed:ped,self.node_ind:np.array(range(self.nnode)),self.Y_target:self.log_Path_RWR[pst:ped,:]})#npath*L
				fout= open('result/PathwayEmb/2D_visualization/all_pathway.txt'+str(epoch),'w')
				for i in range(self.npath):
					fout.write('sb' + '\t')
					for j in range(2):
						fout.write(str(path_mu[i,j])+'\t')
					fout.write('\n')
				fout.close()
			pst = 0
			ped = self.npath
			if epoch == self.max_iter - 1:
				final_path_cov[epoch] = []#sess.run(self.cov_inv_k,feed_dict={self.path_slice_st:pst,self.path_slice_ed:ped,self.node_ind:np.array(range(self.nnode)),self.Y_target:self.log_Path_RWR[pst:ped,:]})#npath*L*L
				final_path_mu[epoch] = []#sess.run(self.path_mu,feed_dict={self.path_slice_st:pst,self.path_slice_ed:ped,self.node_ind:np.array(range(self.nnode)),self.Y_target:self.log_Path_RWR[pst:ped,:]})#npath*L
				final_g2g_node_emb[epoch] = []#sess.run(self.node_mu,feed_dict={self.path_slice_st:pst,self.path_slice_ed:ped,self.node_ind:np.array(range(self.nnode)),self.Y_target:self.log_Path_RWR[pst:ped,:]})#npath*L
				final_p2g[epoch] = np.zeros((self.npath, self.nnode))
				for pi in range(0,self.npath,self.path_batch_size):
					pst = pi
					ped = pi + self.path_batch_size
					ped = min(ped,self.npath)
					final_p2g[epoch][pst:ped,:] = sess.run(self.p2g,feed_dict={self.path_slice_st:pst,self.path_slice_ed:ped,self.node_ind:np.array(range(self.nnode)),self.Y_target:self.log_Path_RWR[pst:ped,:]})#npath*L

		tf.reset_default_graph()
		sess.close()
		return final_path_mu, final_path_cov, final_g2g_node_emb, final_p2g

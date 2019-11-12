import numpy as np
import tensorflow as tf
from .utils import *
#from tensorflow_probability import distributions as tfd
import sys
from scipy.optimize import minimize
from scipy import stats

class Graph2Gauss:
	def __init__(self, log_Path_RWR, log_node_RWR, Path_RWR, Path_mat, path2gene,auc_d, node_emb, p2i, path2label, L, alpha, optimize_gene_vec=False, lr = 1e-2, K=1, p_val=0.10, p_test=0.05, p_nodes=0.0, n_hidden=None,
				 max_iter=2000, tolerance=1000, scale=False, seed=0, verbose=True):
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
		self.auc_d = auc_d
		self.alpha = alpha
		self.lr = lr
		self.path2gene = path2gene
		self.optimize_gene_vec = optimize_gene_vec
		self.Path_RWR = Path_RWR.astype(np.float32)
		self.Path_mat = Path_mat.astype(np.float32)
		np.random.seed(seed)
		self.nselect_path = np.shape(log_Path_RWR)[0]
		self.p2i = p2i
		self.path2label = path2label
		self.log_Path_RWR = log_Path_RWR.astype(np.float32)
		self.log_node_RWR = log_node_RWR.astype(np.float32)
		self.node_emb = node_emb.astype(np.float32)
		#self.path_mu_init = path_mu_init.astype(np.float32)
		#print 'erere',np.shape(self.node_emb)
		# completely hide some nodes from the network for inductive evaluation

		self.node_emb = tf.convert_to_tensor(self.node_emb)#tf.SparseTensor(*sparse_feeder(self.node_emb))
		self.feed_dict = None
		#print 'ere222re',self.node_emb.get_shape().as_list()
		self.npath, self.nnode = self.log_Path_RWR.shape
		self.D = self.node_emb.shape[1]
		self.L = L
		self.reg_mu_init = 0
		self.max_iter = max_iter
		self.tolerance = tolerance
		self.scale = scale
		self.verbose = verbose

		if n_hidden is None:
			n_hidden = [512]
		self.n_hidden = n_hidden


		self.__build()
		print 'build finished'
		sys.stdout.flush()
		#self.__dataset_generator(hops, scale_terms)
		self.__build_loss()
		print 'build loss finished'
		sys.stdout.flush()
		# setup the validation set for easy evaluation
		if p_val > 0:
			val_edges = np.row_stack((val_ones, val_zeros))
			self.neg_val_energy = -self.energy_kl(val_edges)
			self.val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1
			self.val_early_stopping = True
		else:
			self.val_early_stopping = False

		# setup the test set for easy evaluation
		if p_test > 0:
			test_edges = np.row_stack((test_ones, test_zeros))
			self.neg_test_energy = -self.energy_kl(test_edges)
			self.test_ground_truth = A[test_edges[:, 0], test_edges[:, 1]].A1

		# setup the inductive test set for easy evaluation
		if p_nodes > 0:
			self.neg_ind_energy = -self.energy_kl(self.ind_pairs)

	def __build(self):
		w_init = tf.contrib.layers.xavier_initializer
		self.node_mu = tf.get_variable(name='node_mu', dtype=tf.float32, shape=[self.nnode, self.L],initializer= w_init())
		self.node_context = tf.get_variable(name='node_context', shape=[self.nnode, self.L], dtype=tf.float32, initializer = w_init())
		self.path_cov_w = tf.get_variable(name='Path_sigma_w', dtype=tf.float32, shape=[self.npath, self.L,  self.n_hidden[0]],initializer= w_init())
		self.path_cov_x = tf.get_variable(name='Path_sigma_x', dtype=tf.float32, shape=[self.npath, self.L,  self.n_hidden[0]],initializer= w_init())
		self.path_mu = tf.get_variable(name='Path_mu',dtype=tf.float32, shape=[self.npath, self.L], initializer= w_init())


	def fast_mf(self):
		#x_cov = tf.nn.relu(tf.matmul(self.path_cov, tf.transpose(self.node_mu)))
		pmat_fast = []
		for p in range(self.npath):
		#	if p%100==0:
		#		print p*1.0/self.npath
			#pmat_fast.append(self.fast_pdf(self.path_mu[p,:], self.path_sigma_square[p,:], self.node_mu)[0])
			#pmat_fast.append(self.fast_full_cov_mat(p, tf.gather(self.path_cov,p)))
			pmat_fast.append(self.fast_full_cov_mat(p))
		x_cov = tf.stack(pmat_fast)
		#tf.linalg.diag(tf.square(tf.math.reciprocal(scale)))
		return x_cov

	def fast_full_cov_mat(self, p):
		#inv_cov = tf.linalg.diag(inv_cov)
		path_bar = tf.gather(self.path_mu, p)
		path_bar = tf.expand_dims(path_bar,0) # change mu to 1 * ndim
		multiply = tf.constant([self.nnode, 1])
		node_bar = tf.tile(path_bar, multiply) # path mu is npath*ndim

		self.x_diff = tf.square(tf.subtract(self.node_mu, node_bar))

		inv_scale_w = tf.gather(self.path_cov_w,p)
		inv_scale_x = tf.gather(self.path_cov_x,p)
		inv_scale = tf.matmul(inv_scale_w, tf.transpose(inv_scale_x))
		cov =tf.matmul(inv_scale, tf.transpose(inv_scale))
		#inv_cov = tf.linalg.inv(cov)
		inv_cov = cov
		#inv_cov = tf.linalg.diag((inv_scale))
		x_cov = tf.matmul(self.x_diff, inv_cov)
		#x_cov_x = tf.multiply(x_cov, self.x_diff)
		x_cov_x_sum = tf.reduce_sum(x_cov, axis =1) * -0.5
		return x_cov_x_sum

	def node_vec_estimation(self):
		gmat = tf.matmul(self.node_mu, tf.transpose(self.node_context))
		return gmat


	def __build_loss(self):

		self.Path_our = self.fast_mf()
		if self.optimize_gene_vec:
			self.gmat = self.node_vec_estimation()
			self.loss = tf.losses.mean_squared_error(self.Path_our, tf.convert_to_tensor(self.log_Path_RWR)) +tf.losses.mean_squared_error(self.gmat, tf.convert_to_tensor(self.log_node_RWR))

		else:
			self.loss = tf.losses.mean_squared_error(self.Path_our, tf.convert_to_tensor(self.log_Path_RWR))
			print 'here'



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

		sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=gpu_list,
																		  allow_growth=True)))
		sess.run(tf.global_variables_initializer())
		self.best_para = {}



		for epoch in range(self.max_iter):

			Path_our = self.fast_mf()
			self.best_para['node_mu'] = sess.run(self.node_mu)
			self.best_para['node_context'] = sess.run(self.node_context)
			self.best_para['path_cov_w'] = sess.run(self.path_cov_w)
			self.best_para['path_cov_x'] = sess.run(self.path_cov_x)
			self.best_para['path_mu'] = sess.run(self.path_mu)

			Path_our = sess.run(Path_our)

			path_cov_w = self.best_para['path_cov_w']
			path_cov_x = self.best_para['path_cov_x']
			path_mu = self.best_para['path_mu']
			for p in range(2):
				cov_w = path_cov_w[p,:]
				cov_x = path_cov_x[p,:]
				inv_scale = np.dot(cov_w, cov_x.T)
				prec =np.dot(inv_scale, inv_scale.T)
				cov = inv(prec)
				print p, prec, cov

			low_l = [1,11,51,1]
			up_l = [10,50,1000,1000]
			for ii,low in enumerate(low_l):
				auc, auc_std, auc_d, auprc,best_rank = evalute_path_emb(self.path2label, Path_our, self.p2i, self.nselect_path,up=up_l[ii],path2gene=self.path2gene,low=low)
			print
			auc, auc_std, auc_d, auprc,best_rank = evalute_path_emb(self.path2label, Path_our, self.p2i, self.nselect_path)
			m2pv = {}
			pv = -1
			for method in self.auc_d:
				auc_our = []
				auc_base = []
				for d in self.auc_d[method]:
					auc_base.append(self.auc_d[method][d])
					auc_our.append(auc_d[d])
				pv = stats.wilcoxon(auc_our, auc_base)[1]
				if np.mean(auc_our) <= np.mean(auc_base):
					pv = 1.
				m2pv[method] = pv


			if auc > max_auc:
				max_auc = auc
				max_std = auc_std
				max_epoch = epoch
				best_pv = pv

			print epoch,'auc',auc,'best auc',max_auc,auprc,np.min(best_rank),max_std,best_pv,max_epoch,
			for method in self.auc_d:
				print method+':'+'%.2E' % m2pv[method],
			print
			sys.stdout.flush()

			loss, _ = sess.run([self.loss, train_op], self.feed_dict)

			if self.val_early_stopping:
				val_auc, val_ap = score_link_prediction(self.val_ground_truth, sess.run(self.neg_val_energy, self.feed_dict))
				early_stopping_score = val_auc + val_ap

				if self.verbose and epoch % 50 == 0:
					print('epoch: {:3d}, loss: {:.10f}, val_auc: {:.4f}, val_ap: {:.4f}'.format(epoch, loss, val_auc, val_ap))

			else:
				early_stopping_score = -loss
				if self.verbose and epoch % 1 == 0:
					print('epoch: {:3d}, loss: {:.10f}'.format(epoch, loss))

			if early_stopping_score > early_stopping_score_max:
				early_stopping_score_max = early_stopping_score
				tolerance = self.tolerance
				self.__save_vars(sess)
			else:
				tolerance -= 1

			if tolerance == 0:
				break

		if tolerance > 0:
			print('WARNING: Training might not have converged. Try increasing max_iter')

		#self.__restore_vars(sess)


		return Path_our



'''

		pi = 1
		gi = 3
		path_bar = sess.run(tf.gather(self.path_mu, pi))
		x_diff = sess.run(tf.gather(self.node_mu, gi))
		cov = sess.run(tf.gather(cov_inv_k, pi))
		x_diff = x_diff - path_bar
		s = np.dot(x_diff, cov)
		print s
		s = np.dot(np.dot(x_diff, cov), x_diff.T)
		print s
		sys.exit(-1)

			pmat_fast.append(self.fast_full_cov_mat_new(p,cov_qk))
			if p%2000==0:
				process = psutil.Process(os.getpid())
				s1 = time.time() - start_time
				print p,s1,process.memory_info().rss/1024./1024./1024.,'GB'
				GPUtil.showUtilization()

		#inv_scale_qk = sess.run(inv_scale_qk)
		#cov_qk = sess.run(cov_qk)
		#GPUtil.showUtilization()

		pmat_fast = []
		start_time = time.time()
		for p in range(self.npath):
			inv_scale_w = tf.gather(self.path_cov_w,p)
			inv_scale_x_trans = tf.gather(self.path_cov_x,p)
			inv_scale = tf.matmul(inv_scale_w, tf.transpose(inv_scale_x_trans))
			#cov =tf.matmul(inv_scale, tf.transpose(inv_scale))
			cov =tf.matmul(inv_scale, tf.transpose(inv_scale))
			pmat_fast.append(cov)
			s4 = time.time() - start_time
			print p,self.npath,s3,s4,inv_scale.get_shape()#,inv_scale_qk.get_shape()



		pmat_fast = tf.stack(pmat_fast)
		pmat_fast = sess.run(pmat_fast)
		print cov_qk - pmat_fast
		print np.array_equal(cov_qk, pmat_fast)
		#print pmat_fast
		#print tf.equal(pmat_fast, inv_scale_qk)
		sys.exit(-1)
		for p in range(self.npath):
			pmat_fast.append(self.fast_full_cov_mat(p))
			if p%100==0:
				GPUtil.showUtilization()



		x_cov = tf.stack(pmat_fast)

		print self.nnode,self.npath,self.L,self.n_hidden
		path_bar = tf.expand_dims(self.node_mu,0) # change mu to 1 * ndim
		multiply = tf.constant([self.npath, 1, 1])
		node_mu_path = tf.tile(path_bar, multiply) # node_mu_path is npath*nnode*ndim
		print node_mu_path.get_shape()
		print sess.run(node_mu_path)

		cx = tf.matmul(node_mu_path,inv_scale_qk)
		print cx.get_shape()
		print sess.run(cx)

		cx = tf.reshape(cx, [self.nnode * self.npath, self.L])
		cx = tf.reduce_sum( tf.multiply( cx, cx), 1, keep_dims=True )
		print cx.get_shape()
		print sess.run(cx)

		cx = tf.reshape(cx, [self.npath,self.nnode])
		xsigx = tf.transpose(cx)
		print cx.get_shape()
		print sess.run(cx)

		for p in range(self.npath):
			print p
			inv_c = tf.gather(inv_scale_qk, p)
			xc = tf.matmul(self.node_mu, inv_c)
			xc = tf.reduce_sum( tf.multiply( xc, xc ), 1, keep_dims=True )
			data = sess.run(xc)
			print data

		path_mu_ext = tf.expand_dims(self.path_mu,2) # change mu to 1 * ndim
		print path_mu_ext.get_shape(),inv_cov_qk.get_shape()
		xsigmu = tf.matmul(inv_cov_qk, path_mu_ext)
		print xsigmu.get_shape()
		print sess.run(path_mu_ext)

		path_mu_ext = tf.expand_dims(self.path_mu,2) # change mu to 1 * ndim
		print path_mu_ext.get_shape(),inv_cov_qk.get_shape()
		xsigmu = tf.matmul(inv_cov_qk, path_mu_ext)
		print xsigmu.get_shape()
		print sess.run(path_mu_ext)

		sys.exit(-1)

		var = tf.gather(cov_qk, 1)
		print var.get_shape()

		start_time = time.time()



		xvarx = tf.matmul(tf.matmul(self.node_mu,var),tf.transpose(self.node_mu))
		xvarm = tf.matmul(tf.matmul(self.node_mu,var),tf.transpose(self.path_mu))
		mvarx = tf.matmul(tf.matmul(self.path_mu,var),tf.transpose(self.node_mu))
		mvarm = tf.matmul(tf.matmul(self.path_mu,var),tf.transpose(self.path_mu))
		print xvarx.get_shape()
		print xvarm.get_shape()
		print mvarx.get_shape()
		print mvarm.get_shape()

		#path_our = tf.zeros([self.npath, self.nnode])
		#for i in range(self.npath):
		#	for j in range(100):
		#		path_our = xvarx[j,j] - mvarx[i,j] + mvarm[i,i] - xvarm[j,i]
		s2 = time.time() - start_time
		print 's2',s2

		var = tf.gather(cov_qk, 1)
		var = sess.run(var)
		node_mu = sess.run(self.node_mu)
		path_mu = sess.run(self.path_mu)
		start_time = time.time()

		xvarx = np.dot(np.dot(node_mu,var),node_mu.T)
		xvarm = np.dot(np.dot(node_mu,var),path_mu.T)
		mvarx = np.dot(np.dot(path_mu,var),node_mu.T)
		mvarm = np.dot(np.dot(path_mu,var), path_mu.T)
		#s3 = time.time() - start_time
		#path_our = np.zeros((self.npath, self.nnode))
		#for i in range(self.npath):
		#	for j in range(100):
		#		path_our = xvarx[i,i] - mvarx[j,i] + mvarm[j,j] - xvarm[i,j]
		s3 = time.time() - start_time
		print 's3',s3
		return path_our


		path_bar = tf.gather(self.path_mu, p)
		path_bar = tf.expand_dims(path_bar,0) # change mu to 1 * ndim
		multiply = tf.constant([self.nnode, 1])
		node_bar = tf.tile(path_bar, multiply) # path mu is npath*ndim

		start_time = time.time()
		self.x_diff = tf.square(tf.subtract(self.node_mu, node_bar))

		start_time = time.time()
		for p in range(self.npath):
			pmat_fast.append(self.fast_full_cov_mat(p,cov_qk))
			if p%2000==0:
				process = psutil.Process(os.getpid())
				s1 = time.time() - start_time
				print p,s1,process.memory_info().rss/1024./1024./1024.,'GB'
				GPUtil.showUtilization()
'''


		#sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0',allow_growth=True)))
		#sess.run(tf.global_variables_initializer())
		'''
		start_time = time.time()
		pgd = []
		#cov_agg = []
		for p in range(self.npath):
			path_bar = tf.gather(self.path_mu, p)
			path_bar = tf.expand_dims(path_bar,0) # change mu to 1 * ndim
			multiply = tf.constant([self.nnode, 1])
			node_bar = tf.tile(path_bar, multiply) # path mu is npath*ndim
			x_diff = tf.subtract(self.node_mu, node_bar)
			pgd.append(x_diff)
			#cov_agg.append(tf.gather(cov_inv_k, p))
			#if p%10==0:
			#	process = psutil.Process(os.getpid())
			#	s1 = time.time() - start_time
			#	print p,s1,process.memory_info().rss/1024./1024./1024.,'GB'
			#	GPUtil.showUtilization()
		#cov_agg = cov_inv_k
		#if p%10==0:
		process = psutil.Process(os.getpid())
		s1 = time.time() - start_time
		print s1,process.memory_info().rss/1024./1024./1024.,'GB'
		GPUtil.showUtilization()
		start_time = time.time()
		pgd = tf.stack(pgd)
		pdd = tf.stack(cov_inv_k)
		process = psutil.Process(os.getpid())
		s1 = time.time() - start_time
		print s1,process.memory_info().rss/1024./1024./1024.,'GB'
		GPUtil.showUtilization()
		pgd_pdd =  tf.matmul(pgd, pdd)
		process = psutil.Process(os.getpid())
		s1 = time.time() - start_time
		print s1,process.memory_info().rss/1024./1024./1024.,'GB'
		GPUtil.showUtilization()
		s1 = time.time() - start_time
		print '2',s1
		start_time = time.time()
		tmp_sum = tf.reduce_sum( tf.multiply( pgd, pgd_pdd ), 2, keep_dims=False )
		s1 = time.time() - start_time
		'''

		'''
		path_cov_x_batch =  tf.placeholder(shape=[None, self.L,  self.n_hidden[0]], dtype=tf.float32)
		path_cov_w_batch =  tf.placeholder(shape=[None, self.L,  self.n_hidden[0]], dtype=tf.float32)
		path_mu_batch =  tf.placeholder(shape=[None, self.L], dtype=tf.float32)
		Y_target = tf.placeholder(shape=[None, self.nnode], dtype=tf.float32)

		start_time = time.time()
		path_cov_x_trans = tf.transpose(self.path_cov_x, perm=[0,2,1])
		C_k = tf.matmul(self.path_cov_w, path_cov_x_trans)
		C_k_trans = tf.transpose(C_k, perm=[0,2,1])
		cov_inv_k =  tf.matmul(C_k, C_k_trans)
		s1 = time.time() - start_time


		batch_size = 2000
		start_time = time.time()
		p2g = []
		for bi in range(0,self.npath,batch_size):
			st = bi
			ed = bi + batch_size
			ed = min(ed, self.npath)
			pgd = []
			cov_agg = []
			print st,ed
			for p in range(st,ed):
				path_bar = tf.gather(self.path_mu, p)
				path_bar = tf.expand_dims(path_bar,0) # change mu to 1 * ndim
				multiply = tf.constant([self.nnode, 1])
				node_bar = tf.tile(path_bar, multiply) # path mu is npath*ndim
				x_diff = tf.subtract(self.node_mu, node_bar)
				pgd.append(x_diff)
				cov_agg.append(tf.gather(cov_inv_k, p))
			pgd = tf.stack(pgd)
			pdd = tf.stack(cov_agg)
			pgd_pdd =  tf.matmul(pgd, pdd)
			print pgd.get_shape(),pgd_pdd.get_shape()
			tmp_sum = tf.reduce_sum( tf.multiply( pgd, pgd_pdd ), 2, keep_dims=False )
			p2g.append(tmp_sum)

		p2g = tf.concat(p2g,axis=0)
		print p2g.get_shape()
		process = psutil.Process(os.getpid())
		print 'CPU',process.memory_info().rss/1024./1024./1024.,'GB'
		GPUtil.showUtilization()

		#self.Path_our = p2g
		self.loss = tf.losses.mean_squared_error(p2g, Y_target)
		'''

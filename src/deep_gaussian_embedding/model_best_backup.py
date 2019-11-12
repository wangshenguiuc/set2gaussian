import numpy as np
import tensorflow as tf
from .utils import *
from tensorflow_probability import distributions as tfd
import sys


class Graph2Gauss:
	def __init__(self, log_Path_RWR, Path_RWR, Path_mat, node_emb, p2i, path2label, L, alpha, lr = 1e-2, K=1, p_val=0.10, p_test=0.05, p_nodes=0.0, n_hidden=None,
				 max_iter=2000, tolerance=100, scale=False, seed=0, verbose=True):
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
		self.alpha = alpha
		self.lr = lr
		self.Path_RWR = Path_RWR.astype(np.float32)
		self.Path_mat = Path_mat.astype(np.float32)
		np.random.seed(seed)
		self.nselect_path = np.shape(log_Path_RWR)[0]
		self.p2i = p2i
		self.path2label = path2label
		self.log_Path_RWR = log_Path_RWR.astype(np.float32)
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

		# hold out some validation and/or test edges
		# pre-compute the hops for each node for more efficient sampling
		'''
		if p_val + p_test > 0:
			train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(
				A=A, p_val=p_val, p_test=p_test, seed=seed, neg_mul=1, every_node=True, connected=False,
				undirected=(A != A.T).nnz == 0)
			A_train = edges_to_sparse(train_ones, self.N)
			hops = get_hops(A_train, K)
		else:
			hops = get_hops(A, K)

		scale_terms = {h if h != -1 else max(hops.keys()) + 1:
						   hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
					   for h in hops}
		'''
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

		sizes = [self.D] + self.n_hidden

		for i in range(1, len(sizes)):
			W = tf.get_variable(name='W{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32,
								initializer=w_init())
			b = tf.get_variable(name='b{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer = w_init())

			if i == 1:
				encoded = tf.matmul(self.node_emb, W) + b
			else:
				encoded = tf.matmul(encoded, W) + b

			encoded = tf.nn.relu(encoded)
		W_mu = tf.get_variable(name='W_mu', shape=[sizes[-1], self.L], dtype=tf.float32, initializer = w_init())
		b_mu = tf.get_variable(name='b_mu', shape=[self.L], dtype=tf.float32, initializer=w_init())
		self.node_mu = tf.matmul(encoded, W_mu) + b_mu
		#self.node_mu = self.node_emb
		#self.path_mu_init = tf.matmul(self.Path_RWR, self.node_mu)
		#self.node_mu = self.node_emb#, shape=[self.npath, self.L]
		self.path_cov = tf.get_variable(name='Path_sigma', dtype=tf.float32, shape=[self.npath, self.L,  self.L],initializer= w_init())
		#self.path_mu = tf.get_variable(name='Path_mu',dtype=tf.float32, shape=[self.npath, self.L], initializer= w_init())
		#self.path_mu = tf.get_variable(name='Path_mu',dtype=tf.float32, initializer= self.path_mu_init)# shape=[self.npath, self.L],
		#self.path_mu = self.path_mu_init# shape=[self.npath, self.L],

	def guassian_prob(self, pi, gi):
		mvn = tfd.MultivariateNormalDiag(
		loc = tf.gather(self.path_mu, pi),
		scale_diag= tf.gather(self.path_sigma_square, pi))
		gvec = tf.gather(self.node_mu, gi)
		#print mvn.mean().eval()
		#print mvn.stddev().eval()
		#print gvec

		prob = mvn.prob(gvec)
		sys.stdout.flush()
		return prob

	def kl_divergence(self, prob1, prob2):
		'''
		X = tf.distributions.Categorical(probs=x)

		ij_mu = tf.gather(self.mu, pairs)
		ij_sigma = tf.gather(self.sigma, pairs)

		sigma_ratio = ij_sigma[:, 1] / ij_sigma[:, 0]
		trace_fac = tf.reduce_sum(sigma_ratio, 1)
		log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

		mu_diff_sq = tf.reduce_sum(tf.square(ij_mu[:, 0] - ij_mu[:, 1]) / ij_sigma[:, 0], 1)
		'''
		loss = tf.nn.l2_loss(prob1 - prob2)
		return loss

	def direct_infer(self):
		self.path_mu = tf.matmul(tf.convert_to_tensor(self.Path_RWR), self.node_mu)



		path_sigma = []
		for p in range(self.npath):
			coef = tf.constant(self.Path_RWR[p,:])
			coef = tf.expand_dims(coef,0)
			path_bar = tf.gather(self.path_mu, p)
			path_bar = tf.expand_dims(path_bar,0) # change mu to 1 * ndim
			multiply = tf.constant([self.nnode, 1])
			node_bar = tf.tile(path_bar, multiply) # path mu is npath*ndim

			x_diff = tf.square(tf.subtract(self.node_mu, node_bar))
			path_sigma.append( tf.sqrt(tf.squeeze(tf.matmul(coef, x_diff),axis=0)))

		self.path_sigma_square = tf.stack(path_sigma)
		print self.path_sigma_square.get_shape().as_list()



	def closed_form_esitmation(self):
		self.path_mu = tf.matmul(tf.convert_to_tensor(self.Path_RWR), self.node_mu)
		path_sigma_square_l = []
		for p in range(self.npath):
			coef = tf.constant(self.Path_RWR[p,:])
			coef = tf.expand_dims(coef,0)
			path_mu_p = tf.gather(self.path_mu, p)
			path_mu_p = tf.expand_dims(path_mu_p,0) # change mu to 1 * ndim

			multiply = tf.constant([self.nnode, 1])
			node_mu_avg = tf.tile(path_mu_p, multiply) # path mu is npath*ndim
			x_diff = tf.subtract(self.node_mu, node_mu_avg) # x_diff is ngene*ndim
			x_diff = tf.multiply(x_diff,x_diff)
			sigma_square = tf.squeeze(tf.matmul(coef, x_diff),axis=0)
			path_sigma_square_l.append(sigma_square)
			#path_sigma_l.append( tf.sqrt(tf.squeeze(tf.matmul(coef, x_diff),axis=0)))
		self.path_sigma_square = tf.stack(path_sigma_square_l)# this is sigma square
		return self.path_mu, self.path_sigma_square


	def fast_full_cov(self, p, inv_cov):
		path_bar = tf.gather(self.path_mu, p)
		path_bar = tf.expand_dims(path_bar,0) # change mu to 1 * ndim
		multiply = tf.constant([self.nnode, 1])
		node_bar = tf.tile(path_bar, multiply) # path mu is npath*ndim

		self.x_diff = tf.square(tf.subtract(self.node_mu, node_bar))

		x_cov = tf.matmul(self.x_diff, inv_cov)
		x_cov_x = tf.multiply(x_cov, self.x_diff)
		x_cov_x_sum = tf.reduce_sum(x_cov_x, axis =1) * -0.5
		return x_cov_x_sum

		'''
		path_sigma = []
		for p in range(self.npath):
			coef = tf.constant(self.Path_RWR[p,:])
			coef = tf.expand_dims(coef,0)
			path_bar = tf.gather(self.path_mu, p)
			path_bar = tf.expand_dims(path_bar,0) # change mu to 1 * ndim



			path_sigma.append( tf.sqrt(tf.squeeze(tf.matmul(coef, x_diff),axis=0)))

		self.path_sigma_square = tf.stack(path_sigma)
		print self.path_sigma_square.get_shape().as_list()

		#y = inv(scale) @ (x - loc),
		'''
	def density_esitmation(self, p):
		mu = tf.gather(self.path_mu, p)
		sigma_square = tf.gather(self.path_sigma_square, p)
		scale = tf.sqrt(sigma_square)
		prob = self.fast_pdf(mu, scale, self.node_mu)[0]
		return prob



	def fast_pdf(self, mu, scale, X):
		inv_scale = tf.math.reciprocal(scale)

		mu = tf.expand_dims(mu,0) # change mu to 1 * ndim
		multiply = tf.constant([self.nnode, 1])
		mu = tf.tile(mu, multiply) # X is ngene * ndim, make mu ngene * ndim
		mu_diff = tf.subtract(X, mu)
		y = tf.matmul(mu_diff ,tf.diag(inv_scale)) #ngene*ndim multiply  ndim*ndim
		norm_y = tf.square(tf.norm(y, axis=1)) #y is ngene*ndim, norm_y is ngene
		prob_log = tf.scalar_mul(-0.5, norm_y)
		prob_log_min = tf.reduce_max(prob_log)
		prob_log = prob_log - prob_log_min
		prob = tf.exp(prob_log)


		prob_sum = tf.reduce_sum(prob, 0)
		#alt_prob = tf.div(prob, prob_sum)
		#alt_prob_sum = tf.reduce_sum(alt_prob, 0)
		#log_alt_prob = tf.log(alt_prob + self.alpha) - np.log(self.alpha)
		prob = prob + self.alpha * prob_sum
		prob_log = tf.log(prob)
		log_prob_sum = tf.log(prob_sum)


		#print tf.is_nan(prob_sum)
		#prob_norm = tf.div(prob, prob_sum)

		log_prob_norm = prob_log - log_prob_sum - np.log(self.alpha)

		return log_prob_norm,prob_log,y,y,y,y
		#y = inv(scale) @ (x - loc),



	def __build_loss(self):
		self.direct_infer()
		self.path_mu = tf.matmul(tf.convert_to_tensor(self.Path_RWR), self.node_mu)
		pmat_fast = []
		for p in range(self.npath):
			#pmat_fast.append(self.fast_pdf(self.path_mu[p,:], self.path_sigma_square[p,:], self.node_mu)[0])
			pmat_fast.append(self.fast_full_cov(p, tf.gather(self.path_cov,p)))
		Path_our = tf.stack(pmat_fast)
		self.loss = tf.losses.mean_squared_error(Path_our, tf.convert_to_tensor(self.log_Path_RWR))#*self.nnode*self.npath #+ self.reg_mu_init * tf.losses.mean_squared_error(self.path_mu, path_mu_init)

		#Path_our = sess.run(Path_our)
		'''
		hop_pos = tf.stack([self.triplets[:, 0], self.triplets[:, 1]], 1)
		hop_neg = tf.stack([self.triplets[:, 0], self.triplets[:, 2]], 1)
		eng_pos = self.energy_kl(hop_pos)
		eng_neg = self.energy_kl(hop_neg)
		energy = tf.square(eng_pos) + tf.exp(-eng_neg)

		if self.scale:
			self.loss = tf.reduce_mean(energy * self.scale_terms)
		else:
			self.loss = tf.reduce_mean(energy)
		'''

	def __setup_inductive(self, A, X, p_nodes):
		N = A.shape[0]
		nodes_rnd = np.random.permutation(N)
		n_hide = int(N * p_nodes)
		nodes_hide = nodes_rnd[:n_hide]

		A_hidden = A.copy().tolil()
		A_hidden[nodes_hide] = 0
		A_hidden[:, nodes_hide] = 0

		# additionally add any dangling nodes to the hidden ones since we can't learn from them
		nodes_dangling = np.where(A_hidden.sum(0).A1 + A_hidden.sum(1).A1 == 0)[0]
		if len(nodes_dangling) > 0:
			nodes_hide = np.concatenate((nodes_hide, nodes_dangling))
		nodes_keep = np.setdiff1d(np.arange(N), nodes_hide)

		self.node_emb = tf.sparse_placeholder(tf.float32)
		self.feed_dict = {self.node_emb: sparse_feeder(X[nodes_keep])}

		self.ind_pairs = batch_pairs_sample(A, nodes_hide)
		self.ind_ground_truth = A[self.ind_pairs[:, 0], self.ind_pairs[:, 1]].A1
		self.ind_feed_dict = {self.node_emb: sparse_feeder(X)}

		A = A[nodes_keep][:, nodes_keep]

		return A

	def energy_kl(self, pairs):
		"""
		Computes the energy of a set of node pairs as the KL divergence between their respective Gaussian embeddings.

		Parameters
		----------
		pairs : array-like, shape [?, 2]
			The edges/non-edges for which the energy is calculated

		Returns
		-------
		energy : array-like, shape [?]
			The energy of each pair given the currently learned model
		"""
		ij_mu = tf.gather(self.mu, pairs)
		ij_sigma = tf.gather(self.sigma, pairs)

		sigma_ratio = ij_sigma[:, 1] / ij_sigma[:, 0]
		trace_fac = tf.reduce_sum(sigma_ratio, 1)
		log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

		mu_diff_sq = tf.reduce_sum(tf.square(ij_mu[:, 0] - ij_mu[:, 1]) / ij_sigma[:, 0], 1)

		return 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

	def __dataset_generator(self, hops, scale_terms):
		"""
		Generates a set of triplets and associated scaling terms by:
			1. Sampling for each node a set of nodes from each of its neighborhoods
			2. Forming all implied pairwise constraints

		Uses tf.Dataset API to perform the sampling in a separate thread for increased speed.

		Parameters
		----------
		hops : dict
			A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
		scale_terms : dict
			The appropriate up-scaling terms to ensure unbiased estimates for each neighbourhood
		Returns
		-------
		"""
		def gen():
			while True:
				yield to_triplets(sample_all_hops(hops), scale_terms)

		dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.float32), ([None, 3], [None]))
		self.triplets, self.scale_terms = dataset.prefetch(1).make_one_shot_iterator().get_next()

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
		train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

		sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=gpu_list,
																		  allow_growth=True)))
		sess.run(tf.global_variables_initializer())
		for epoch in range(self.max_iter):
			'''
			mu = self.path_mu
			sigma = self.path_sigma_square

			path_mu = tf.expand_dims(mu,1)
			path_sigma = tf.expand_dims(sigma,1)
			node_mu = tf.expand_dims(self.node_mu,0)
			multiply = tf.constant([self.npath, 1, 1])
			node_mu = tf.tile(node_mu, multiply)
			mvn = tfd.MultivariateNormalDiag(
			loc = path_mu ,
			scale_diag= path_sigma)
			Path_our = mvn.prob(node_mu)
			Path_sum_vec = tf.reduce_sum(Path_our, 1)
			Path_sum_vec = tf.expand_dims(Path_sum_vec, 1)
			Path_sum_mat = tf.tile(Path_sum_vec, tf.constant([1, self.nnode]))
			Path_our = tf.div(Path_our, Path_sum_mat)
			#Path_our = tf.scalar_mul(self.nnode, Path_our)
			Path_our = sess.run(Path_our)
			Path_our_log = np.log(Path_our + self.alpha) - np.log(self.alpha)
			'''
			self.path_mu = tf.matmul(tf.convert_to_tensor(self.Path_RWR), self.node_mu)
			pmat_fast = []
			for p in range(self.npath):
				#pmat_fast.append(self.fast_pdf(self.path_mu[p,:], self.path_sigma_square[p,:], self.node_mu)[0])
				pmat_fast.append(self.fast_full_cov(p, tf.gather(self.path_cov,p)))
			Path_our = tf.stack(pmat_fast)
			Path_our = sess.run(Path_our)


			auc = evalute_path_emb(self.path2label, Path_our, self.p2i, self.nselect_path)
			p2p = sess.run(tf.matmul(self.path_mu, tf.transpose(self.path_mu)))
			auc_sim = evalute_path_sim(self.path2label, p2p, self.p2i)
			if auc > max_auc:
				max_auc = auc
				max_epoch = epoch
			if auc_sim > max_auc_sim:
				max_auc_sim = auc_sim
				max_auc_sim_epoch = epoch
			if max_auc - auc > 0.05 and epoch>100:
				break
			print epoch,'auc',auc,auc_sim,'best auc',max_auc,max_epoch,max_auc_sim,max_auc_sim_epoch
			sys.stdout.flush()
			'''
			prob = tf.zeros([self.npath, self.nnode])#np.zeros((self.npath, self.nnode))
			p_mat = []
			path_mu = tf.expand_dims(self.path_mu,1)
			path_sigma = tf.expand_dims(self.path_sigma_square,1)

			node_mu = tf.expand_dims(self.node_mu,0)
			multiply = tf.constant([self.npath, 1, 1])
			#print path_sigma.get_shape().as_list()
			#print multiply.get_shape().as_list()
			#print node_mu.get_shape().as_list()
			node_mu = tf.tile(node_mu, multiply)
			#print x.get_shape().as_list()
			#node_mu = x
			#print node_mu.get_shape().as_list()
			mvn = tfd.MultivariateNormalDiag(
			loc = path_mu ,
			scale_diag= path_sigma)
			#gvec = []
			#for gi in range(self.nnode):
			#	gvec.append([tf.gather(self.node_mu, gi)])
			#	if gi%1000==0:
			#		print gi
			prob = mvn.prob(node_mu)

			print 'slw1',sess.run(prob)

			prob = tf.zeros([self.npath, self.nnode])#np.zeros((self.npath, self.nnode))
			p_mat = []
			for pi in range(self.npath):
				g_l = []
				sum = 0.
				for gi in range(self.nnode):
					g_l.append(self.guassian_prob(pi, gi))
					sum += self.guassian_prob(pi, gi)
				#for i in range(len(g_l)):
				#	g_l[i] /= sum
				g_vec = tf.stack(g_l)
				p_mat.append(g_vec)
			prob = tf.stack(p_mat,axis=0)
			print 'slw2',sess.run(prob)


			prob = tf.zeros([self.npath, self.nnode])#np.zeros((self.npath, self.nnode))
			p_mat = []
			mvn = tfd.MultivariateNormalDiag(
			loc = self.path_mu ,
			scale_diag= self.path_sigma_square)
			for gi in range(self.nnode):
				gvec = tf.gather(self.node_mu, gi)
				prob = mvn.prob(gvec)
				p_mat.append(prob)
			prob = tf.stack(p_mat,axis=1)
			print 'fst',sess.run(prob)
			#print self.node_emb
			print 'sigma',sess.run(self.path_sigma_square),sess.run(self.path_mu)
			#print 'mu',self.path_mu
			#print '----split---'
			#print self.log_Path_RWR
			sys.stdout.flush()
			'''
			loss, _ = sess.run([self.loss, train_op], self.feed_dict)
			'''
			scale = self.path_sigma_square[0,:]
			mu = self.path_mu[0,:]
			X = self.node_mu
			inv_scale = tf.math.reciprocal(scale)

			mu = tf.expand_dims(mu,0) # change mu to 1 * ndim
			multiply = tf.constant([self.nnode, 1])
			mu = tf.tile(mu, multiply) # X is ngene * ndim, make mu ngene * ndim
			mu_diff = tf.subtract(X, mu)
			y = tf.matmul(mu_diff ,tf.diag(inv_scale)) #ngene*ndim multiply  ndim*ndim
			print 'y',sess.run(y)
			norm_y = tf.square(tf.norm(y, axis=1)) #y is ngene*ndim, norm_y is ngene
			print 'norm_y',sess.run(norm_y)
			prob_log = tf.scalar_mul(-0.5, norm_y)
			prob_log_min = tf.reduce_max(prob_log)
			prob_log = prob_log - prob_log_min
			prob = tf.exp(prob_log)


			prob_sum = tf.reduce_sum(prob, 0)

			#alt_prob = tf.div(prob, prob_sum)
			#alt_prob_sum = tf.reduce_sum(alt_prob, 0)
			#log_alt_prob = tf.log(alt_prob + self.alpha) - np.log(self.alpha)
			prob_alpha = prob + self.alpha * prob_sum
			prob_log_new = tf.log(prob_alpha)
			log_prob_sum = tf.log(prob_sum)


			#print tf.is_nan(prob_sum)
			#prob_norm = tf.div(prob, prob_sum)

			log_prob_norm = prob_log_new - log_prob_sum - np.log(self.alpha)
			dc_mu, dc_sigma = tf.gradients(prob, [self.path_mu,self.path_sigma_square])
			print 'grad',sess.run(dc_mu),sess.run(dc_sigma),sess.run(prob),np.min(sess.run(prob)),np.max(sess.run(prob))
			dc_mu, dc_sigma = tf.gradients(prob_log, [self.path_mu,self.path_sigma_square])
			print 'grad',sess.run(dc_mu),sess.run(dc_sigma),sess.run(prob_log),np.min(sess.run(prob_log)),np.max(sess.run(prob_log))
			dc_mu, dc_sigma = tf.gradients(prob_alpha, [self.path_mu,self.path_sigma_square])
			print 'grad',sess.run(dc_mu),sess.run(dc_sigma)
			'''
			if self.val_early_stopping:
				val_auc, val_ap = score_link_prediction(self.val_ground_truth, sess.run(self.neg_val_energy, self.feed_dict))
				early_stopping_score = val_auc + val_ap

				if self.verbose and epoch % 50 == 0:
					print('epoch: {:3d}, loss: {:.4f}, val_auc: {:.4f}, val_ap: {:.4f}'.format(epoch, loss, val_auc, val_ap))

			else:
				early_stopping_score = -loss
				if self.verbose and epoch % 1 == 0:
					print('epoch: {:3d}, loss: {:.4f}'.format(epoch, loss))

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

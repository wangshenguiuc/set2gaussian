import numpy as np
import scipy.sparse as sp
import warnings
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import normalize
from scipy.interpolate import griddata
import numpy.ma as ma
from numpy.random import uniform, seed
from matplotlib import cm
from numpy import linalg as LA
import copy
import math
import scipy.spatial as sp


def evaluate_vec(pred,truth): #auc,pear,spear,auprc,prec_at_k,f1=evaluate_vec(pred,truth)
	pred = np.array(pred)
	truth = np.array(truth)


	if set(np.unique(truth))==set([0,1]):
		fpr, tpr, thresholds = metrics.roc_curve(truth, pred, pos_label=1)
		auc = metrics.auc(fpr, tpr)
		auprc = metrics.average_precision_score(truth, pred)
		if set(np.unique(pred))==set([0,1]):
			f1 =  f1_score(truth, pred, average='macro')
			acc = metrics.accuracy_score(truth, pred)
		else:
			f1 = 0.
			acc = 0.
		pear = 0.
		spear = 0.
		prec_at_k = precision_at_k(pred,truth)
	else:
		auc = 0.5
		auprc = 0.
		f1 = 0.
		acc = 0.
		pear = scipy.stats.pearsonr(pred, truth)[0]
		spear = scipy.stats.spearmanr(pred, truth)[0]
		prec_at_k = 0.
	return auc,pear,spear,auprc,prec_at_k,f1,acc

def evaluate_pathway_member(p2g_score, p2g_truth_test,p2g_truth_train, low_b=[3,11,31],up_b=[10,30,1000]):
	#, low_b=[1,4,7,11],up_b=[3,6,10,50]
	p2g_score_new = copy.deepcopy(p2g_score) * -1
	npath,ngene = np.shape(p2g_score_new)
	nbin = len(low_b)
	auroc_l = {}
	for b in range(nbin):
		auroc_l[b] = []
	prec_l = {}
	for b in range(nbin):
		prec_l[b] = []
	p2g_truth = p2g_truth_test + p2g_truth_train
	for i in range(npath):
		predict_ind1 = set(np.where(p2g_truth_train[i,:]==0)[0])
		predict_ind2 = set(np.where(p2g_truth_test[i,:]==1)[0])
		predict_ind = list(predict_ind1.union( predict_ind2))
		predict_ind = np.array(predict_ind)
		truth = p2g_truth_test[i,predict_ind]
		#truth = p2g_truth_train[i,:]
		sc = p2g_score_new[i,predict_ind]
		nsample = np.sum(truth)
		this_b = -1
		for b in range(nbin):
			if low_b[b] <= nsample and up_b[b] >= nsample:
				this_b = b
				break
		if this_b == -1:
			continue
		_,_,_,auprc,prec_at_k = evaluate_vec(sc,truth)[0:5]
		if np.isnan(auprc):
			print sc, truth
			sys.exit('wrong auprc function')
		auroc_l[this_b].append(auprc)
		prec_l[this_b].append(prec_at_k)
	auroc_d = {}
	prec_d = {}
	for b in auroc_l:
		if len(auroc_l[b]) > 0:
			auroc_d[b] = (np.nanmean(auroc_l[b]), len(auroc_l[b]))
			prec_d[b] = (np.nanmean(prec_l[b]), len(prec_l[b]))
	return auroc_d, auroc_l,prec_d,prec_l

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))) * 180 / 3.14


def plot_gaussian_distr(node_d,path_d,fwrite=''):
	plt.clf()
	fig, ax = plt.subplots()
	up = -1
	down = 1
	for d in node_d:
		ax.plot(node_d[d][0], node_d[d][1], 'ro', zorder=3)
		ax.annotate(d, xy=(node_d[d][0], node_d[d][1]),fontsize=6, zorder=3)
		#up = max(up, no)
	for d in path_d:
		mu,cov = path_d[d]
		ax.plot(mu[0], mu[1], 'bs', zorder=2)
		ax.annotate(d, xy=(mu[0], mu[1]),fontsize=6)
		max_cov = np.max(np.abs(cov))
		cov = cov / max_cov
		x_vec= [1,0]
		_, Evec = LA.eig(cov)
		rotation = angle(x_vec,Evec[:,0]) + 90
		e1 = patches.Ellipse((mu[0], mu[1]), cov[0,0], cov[1,1],
						 angle=rotation, linewidth=2, fill=True, zorder=2)
		ax.add_patch(e1)
		#up_tmp = max(mu[0] + cov[0,0], mu[1]+cov[1,1])
		#down_tmp = max(mu[0] - cov[0,0], mu[1] - cov[1,1])
	#ax.set_xlim([-3,3])
	#ax.set_ylim([-3,3])
	ax.relim()
	ax.autoscale_view()
	fig.savefig(fwrite)

def evalute_path_sim(path2label, p2p, p2i):
	label_l = []
	score_l = []
	for p1 in path2label:
		for p2 in path2label:
			pl = path2label[p1].intersection(path2label[p2])
			if len(pl) == 0:
				label_l.append(0)
			else:
				label_l.append(1)
			score_l.append(p2p[p2i[p2], p2i[p1]])
	auc,pear,spear,auprc = evaluate_vec(score_l,label_l)
	return auc


def evalute_path_emb(path2label, p2g, p2i, nselect_path=1000000,up=100000,low = -1,path2gene=[]):
	auc = {}
	auprc = {}
	best_rank = []
	npath,ngene = np.shape(p2g)
	for path in path2label:
		if p2i[path] >= nselect_path:
			continue
		label_l = path2label[path]
		if len(path2gene) >0 and ( len(path2gene[path])<low or len(path2gene[path])>up):
			continue
		score = np.zeros(ngene)
		label = np.zeros(ngene)
		for g in label_l:
			label[g] = 1
		if np.sum(label) == 0:
			continue
		for i in range(ngene):
			score[i] = p2g[p2i[path], i]
		score_rank = np.argsort(score*-1)
		for g in label_l:
			best_rank.append(np.where(score_rank==g)[0][0])
		auc[path], tmp, tmp, auprc[path] = evaluate_vec(score,label)
	best_rank = np.array(best_rank)
	return np.mean(auc.values()), np.std(auc.values()),auc,np.mean(auprc.values()),best_rank

def edges_to_sparse(edges, N, values=None):
	"""
	Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

	Parameters
	----------
	edges : array-like, shape [n_edges, 2]
		Edge indices
	N : int
		Number of nodes
	values : array_like, shape [n_edges]
		The values to put at the specified edge indices. Optional, default: np.ones(.)

	Returns
	-------
	A : scipy.sparse.csr.csr_matrix
		Sparse adjacency matrix

	"""
	if values is None:
		values = np.ones(edges.shape[0])

	return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
								   every_node=True, connected=False, undirected=False,
								   use_edge_cover=True, set_ops=False, asserts=False):
	"""Split the edges of the adjacency matrix into train, validation and test edges
	and randomly samples equal amount of validation and test non-edges.

	Parameters
	----------
	A : scipy.sparse.spmatrix
		Sparse unweighted adjacency matrix
	p_val : float
		Percentage of validation edges. Default p_val=0.10
	p_test : float
		Percentage of test edges. Default p_test=0.05
	seed : int
		Seed for numpy.random. Default seed=0
	neg_mul : int
		What multiplicity of negative samples (non-edges) to have in the test/validation set
		w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
	every_node : bool
		Make sure each node appears at least once in the train set. Default every_node=True
	connected : bool
		Make sure the training graph is still connected after the split
	undirected : bool
		Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
		Default undirected=False
	use_edge_cover: bool
		Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
		Only active when every_node=True. Default use_edge_cover=True
	set_ops : bool
		Whether to use set operations to construction the test zeros. Default setwise_zeros=True
		Otherwise use a while loop.
	asserts : bool
		Unit test like checks. Default asserts=False

	Returns
	-------
	train_ones : array-like, shape [n_train, 2]
		Indices of the train edges
	val_ones : array-like, shape [n_val, 2]
		Indices of the validation edges
	val_zeros : array-like, shape [n_val, 2]
		Indices of the validation non-edges
	test_ones : array-like, shape [n_test, 2]
		Indices of the test edges
	test_zeros : array-like, shape [n_test, 2]
		Indices of the test non-edges

	"""
	assert p_val + p_test > 0
	assert A.max() == 1  # no weights
	assert A.min() == 0  # no negative edges
	assert A.diagonal().sum() == 0  # no self-loops
	assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes

	is_undirected = (A != A.T).nnz == 0

	if undirected:
		assert is_undirected  # make sure is directed
		A = sp.tril(A).tocsr()  # consider only upper triangular
		A.eliminate_zeros()
	else:
		if is_undirected:
			warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

	np.random.seed(seed)

	E = A.nnz
	N = A.shape[0]
	s_train = int(E * (1 - p_val - p_test))

	idx = np.arange(N)

	# hold some edges so each node appears at least once
	if every_node:
		if connected:
			assert sp.csgraph.connected_components(A)[0] == 1  # make sure original graph is connected
			A_hold = sp.csgraph.minimum_spanning_tree(A)
		else:
			A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
			d = A.sum(1).A1

			if use_edge_cover:
				hold_edges = edge_cover(A)

				# make sure the training percentage is not smaller than len(edge_cover)/E when every_node is set to True
				min_size = hold_edges.shape[0]
				if min_size > s_train:
					raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
									 .format(min_size / E))
			else:
				# make sure the training percentage is not smaller than N/E when every_node is set to True
				if N > s_train:
					raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
									 .format(N / E))

				hold_edges_d1 = np.column_stack(
					(idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

				if np.any(d == 0):
					hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
													 idx[d == 0]))
					hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
				else:
					hold_edges = hold_edges_d1

			if asserts:
				assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
				assert len(np.unique(hold_edges.flatten())) == N

			A_hold = edges_to_sparse(hold_edges, N)

		A_hold[A_hold > 1] = 1
		A_hold.eliminate_zeros()
		A_sample = A - A_hold

		s_train = s_train - A_hold.nnz
	else:
		A_sample = A

	idx_ones = np.random.permutation(A_sample.nnz)

	ones = np.column_stack(A_sample.nonzero())
	train_ones = ones[idx_ones[:s_train]]
	test_ones = ones[idx_ones[s_train:]]

	# return back the held edges
	if every_node:
		train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

	n_test = len(test_ones) * neg_mul

	if set_ops:
		# generate slightly more completely random non-edge indices than needed and discard any that hit an edge
		# much faster compared a while loop
		# in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
		if undirected:
			random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
			random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
		else:
			random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
			random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

		# discard ones
		random_sample = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
		# discard duplicates
		random_sample = random_sample[np.unique(random_sample[:, 0] * N + random_sample[:, 1], return_index=True)[1]]
		# only take as much as needed
		test_zeros = np.row_stack(random_sample)[:n_test]
		assert test_zeros.shape[0] == n_test
	else:
		test_zeros = []
		while len(test_zeros) < n_test:
			i, j = np.random.randint(0, N, 2)
			if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
				test_zeros.append((i, j))
		test_zeros = np.array(test_zeros)

	# split the test set into validation and test set
	s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
	s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

	val_ones = test_ones[:s_val_ones]
	test_ones = test_ones[s_val_ones:]

	val_zeros = test_zeros[:s_val_zeros]
	test_zeros = test_zeros[s_val_zeros:]

	if undirected:
		# put (j, i) edges for every (i, j) edge in the respective sets and form back original A
		symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
		train_ones = symmetrize(train_ones)
		val_ones = symmetrize(val_ones)
		val_zeros = symmetrize(val_zeros)
		test_ones = symmetrize(test_ones)
		test_zeros = symmetrize(test_zeros)
		A = A.maximum(A.T)

	if asserts:
		set_of_train_ones = set(map(tuple, train_ones))
		assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
		assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
		assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
		assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
		assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
		assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
		assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
		assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
		assert not connected or sp.csgraph.connected_components(A_hold)[0] == 1
		assert not every_node or ((A_hold - A) > 0).sum() == 0

	return train_ones, val_ones, val_zeros, test_ones, test_zeros


def sparse_feeder(M):
	"""
	Prepares the input matrix into a format that is easy to feed into tensorflow's SparseTensor

	Parameters
	----------
	M : scipy.sparse.spmatrix
		Matrix to be fed

	Returns
	-------
	indices : array-like, shape [n_edges, 2]
		Indices of the sparse elements
	values : array-like, shape [n_edges]
		Values of the sparse elements
	shape : array-like
		Shape of the matrix
	"""
	M = sp.coo_matrix(M)
	return np.vstack((M.row, M.col)).T, M.data, M.shape


def cartesian_product(x, y):
	"""
	Form the cartesian product (i.e. all pairs of values) between two arrays.
	Parameters
	----------
	x : array-like, shape [Nx]
		Left array in the cartesian product
	y : array-like, shape [Ny]
		Right array in the cartesian product

	Returns
	-------
	xy : array-like, shape [Nx * Ny]
		Cartesian product

	"""
	return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


def score_link_prediction(labels, scores):
	"""
	Calculates the area under the ROC curve and the average precision score.

	Parameters
	----------
	labels : array-like, shape [N]
		The ground truth labels
	scores : array-like, shape [N]
		The (unnormalized) scores of how likely are the instances

	Returns
	-------
	roc_auc : float
		Area under the ROC curve score
	ap : float
		Average precision score
	"""

	return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def score_node_classification(features, z, p_labeled=0.1, n_repeat=10, norm=False):
	"""
	Train a classifier using the node embeddings as features and reports the performance.

	Parameters
	----------
	features : array-like, shape [N, L]
		The features used to train the classifier, i.e. the node embeddings
	z : array-like, shape [N]
		The ground truth labels
	p_labeled : float
		Percentage of nodes to use for training the classifier
	n_repeat : int
		Number of times to repeat the experiment
	norm

	Returns
	-------
	f1_micro: float
		F_1 Score (micro) averaged of n_repeat trials.
	f1_micro : float
		F_1 Score (macro) averaged of n_repeat trials.
	"""
	lrcv = LogisticRegressionCV()

	if norm:
		features = normalize(features)

	trace = []
	for seed in range(n_repeat):
		sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
		split_train, split_test = next(sss.split(features, z))

		lrcv.fit(features[split_train], z[split_train])
		predicted = lrcv.predict(features[split_test])

		f1_micro = f1_score(z[split_test], predicted, average='micro')
		f1_macro = f1_score(z[split_test], predicted, average='macro')

		trace.append((f1_micro, f1_macro))

	return np.array(trace).mean(0)


def get_hops(A, K):
	"""
	Calculates the K-hop neighborhoods of the nodes in a graph.

	Parameters
	----------
	A : scipy.sparse.spmatrix
		The graph represented as a sparse matrix
	K : int
		The maximum hopness to consider.

	Returns
	-------
	hops : dict
		A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
	"""
	hops = {1: A.tolil(), -1: A.tolil()}
	hops[1].setdiag(0)

	for h in range(2, K + 1):
		# compute the next ring
		next_hop = hops[h - 1].dot(A)
		next_hop[next_hop > 0] = 1

		# make sure that we exclude visited n/edges
		for prev_h in range(1, h):
			next_hop -= next_hop.multiply(hops[prev_h])

		next_hop = next_hop.tolil()
		next_hop.setdiag(0)

		hops[h] = next_hop
		hops[-1] += next_hop

	return hops


def sample_last_hop(A, nodes):
	"""
	For each node in nodes samples a single node from their last (K-th) neighborhood.

	Parameters
	----------
	A : scipy.sparse.spmatrix
		Sparse matrix encoding which nodes belong to any of the 1, 2, ..., K-1, neighborhoods of every node
	nodes : array-like, shape [N]
		The nodes to consider

	Returns
	-------
	sampled_nodes : array-like, shape [N]
		The sampled nodes.
	"""
	N = A.shape[0]

	sampled = np.random.randint(0, N, len(nodes))

	nnz = A[nodes, sampled].nonzero()[1]
	while len(nnz) != 0:
		new_sample = np.random.randint(0, N, len(nnz))
		sampled[nnz] = new_sample
		nnz = A[nnz, new_sample].nonzero()[1]

	return sampled


def sample_all_hops(hops, nodes=None):
	"""
	For each node in nodes samples a single node from all of their neighborhoods.

	Parameters
	----------
	hops : dict
		A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
	nodes : array-like, shape [N]
		The nodes to consider

	Returns
	-------
	sampled_nodes : array-like, shape [N, K]
		The sampled nodes.
	"""

	N = hops[1].shape[0]

	if nodes is None:
		nodes = np.arange(N)

	return np.vstack((nodes,
					  np.array([[-1 if len(x) == 0 else np.random.choice(x) for x in hops[h].rows[nodes]]
								for h in hops.keys() if h != -1]),
					  sample_last_hop(hops[-1], nodes)
					  )).T


def to_triplets(sampled_hops, scale_terms):
	"""
	Form all valid triplets (pairwise constraints) from a set of sampled nodes in triplets

	Parameters
	----------
	sampled_hops : array-like, shape [N, K]
	   The sampled nodes.
	scale_terms : dict
		The appropriate up-scaling terms to ensure unbiased estimates for each neighbourhood

	Returns
	-------
	triplets : array-like, shape [?, 3]
	   The transformed triplets.
	"""
	triplets = []
	triplet_scale_terms = []

	for i, j in itertools.combinations(np.arange(1, sampled_hops.shape[1]), 2):
		triplet = sampled_hops[:, [0] + [i, j]]
		triplet = triplet[(triplet[:, 1] != -1) & (triplet[:, 2] != -1)]
		triplet = triplet[(triplet[:, 0] != triplet[:, 1]) & (triplet[:, 0] != triplet[:, 2])]
		triplets.append(triplet)

		triplet_scale_terms.append(scale_terms[i][triplet[:, 1]] * scale_terms[j][triplet[:, 2]])

	return np.row_stack(triplets), np.concatenate(triplet_scale_terms)


def load_dataset(file_name):
	"""Load a graph from a Numpy binary file.

	Parameters
	----------
	file_name : str
		Name of the file to load.

	Returns
	-------
	graph : dict
		Dictionary that contains:
			* 'A' : The adjacency matrix in sparse matrix format
			* 'X' : The attribute matrix in sparse matrix format
			* 'z' : The ground truth class labels
			* Further dictionaries mapping node, class and attribute IDs

	"""
	if not file_name.endswith('.npz'):
		file_name += '.npz'
	with np.load(file_name) as loader:
		loader = dict(loader)
		A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
						   loader['adj_indptr']), shape=loader['adj_shape'])

		X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
						   loader['attr_indptr']), shape=loader['attr_shape'])

		z = loader.get('labels')

		graph = {
			'A': A,
			'X': X,
			'z': z
		}

		idx_to_node = loader.get('idx_to_node')
		if idx_to_node:
			idx_to_node = idx_to_node.tolist()
			graph['idx_to_node'] = idx_to_node

		idx_to_attr = loader.get('idx_to_attr')
		if idx_to_attr:
			idx_to_attr = idx_to_attr.tolist()
			graph['idx_to_attr'] = idx_to_attr

		idx_to_class = loader.get('idx_to_class')
		if idx_to_class:
			idx_to_class = idx_to_class.tolist()
			graph['idx_to_class'] = idx_to_class

		return graph


def edge_cover(A):
	"""
	Approximately compute minimum edge cover.

	Edge cover of a graph is a set of edges such that every vertex of the graph is incident
	to at least one edge of the set. Minimum edge cover is an  edge cover of minimum size.

	Parameters
	----------
	A : sp.spmatrix
		Sparse adjacency matrix

	Returns
	-------
	edges : array-like, shape [?, 2]
		The edges the form the edge cover
	"""

	N = A.shape[0]
	d_in = A.sum(0).A1
	d_out = A.sum(1).A1

	# make sure to include singleton nodes (nodes with one incoming or one outgoing edge)
	one_in = np.where((d_in == 1) & (d_out == 0))[0]
	one_out = np.where((d_in == 0) & (d_out == 1))[0]

	edges = []
	edges.append(np.column_stack((A[:, one_in].argmax(0).A1, one_in)))
	edges.append(np.column_stack((one_out, A[one_out].argmax(1).A1)))
	edges = np.row_stack(edges)

	edge_cover_set = set(map(tuple, edges))
	nodes = set(edges.flatten())

	# greedly add other edges such that both end-point are not yet in the edge_cover_set
	cands = np.column_stack(A.nonzero())
	for u, v in cands[d_in[cands[:, 1]].argsort()]:
		if u not in nodes and v not in nodes and u != v:
			edge_cover_set.add((u, v))
			nodes.add(u)
			nodes.add(v)
		if len(nodes) == N:
			break

	# add a single edge for the rest of the nodes not covered so far
	not_covered = np.setdiff1d(np.arange(N), list(nodes))
	edges = [list(edge_cover_set)]
	not_covered_out = not_covered[d_out[not_covered] > 0]

	if len(not_covered_out) > 0:
		edges.append(np.column_stack((not_covered_out, A[not_covered_out].argmax(1).A1)))

	not_covered_in = not_covered[d_out[not_covered] == 0]
	if len(not_covered_in) > 0:
		edges.append(np.column_stack((A[:, not_covered_in].argmax(0).A1, not_covered_in)))

	edges = np.row_stack(edges)

	# make sure that we've indeed computed an edge_cover
	assert A[edges[:, 0], edges[:, 1]].sum() == len(edges)
	assert len(set(map(tuple, edges))) == len(edges)
	assert len(np.unique(edges)) == N

	return edges


def batch_pairs_sample(A, nodes_hide):
	"""
	For a given set of nodes return all edges and an equal number of randomly sampled non-edges.

	Parameters
	----------
	A : sp.spmatrix
		Sparse adjacency matrix

	Returns
	-------
	pairs : array-like, shape [?, 2]
		The sampled pairs.

	"""
	A = A.copy()
	undiricted = (A != A.T).nnz == 0

	if undiricted:
		A = sp.triu(A, 1).tocsr()

	edges = np.column_stack(A.nonzero())
	edges = edges[np.in1d(edges[:, 0], nodes_hide) | np.in1d(edges[:, 1], nodes_hide)]

	# include the missing direction
	if undiricted:
		edges = np.row_stack((edges, np.column_stack((edges[:, 1], edges[:, 0]))))

	# sample the non-edges for each node separately
	arng = np.arange(A.shape[0])
	not_edges = []
	for nh in nodes_hide:
		nn = np.concatenate((A[nh].nonzero()[1], A[:, nh].nonzero()[0]))
		not_nn = np.setdiff1d(arng, nn)

		not_nn = np.random.permutation(not_nn)[:len(nn)]
		not_edges.append(np.column_stack((np.repeat(nh, len(nn)), not_nn)))

	not_edges = np.row_stack(not_edges)

	# include the missing direction
	if undiricted:
		not_edges = np.row_stack((not_edges, np.column_stack((not_edges[:, 1], not_edges[:, 0]))))

	pairs = np.row_stack((edges, not_edges))

	return pairs

import numpy as np
import collections

from scipy import sparse
class BioNetwork():

	def __init__(self, network_file_l,inter_method='prob',reverse=False,weighted=True):
		self.word_ct = {}
		self.nedge = 0
		self.weighted = weighted
		self.network = collections.defaultdict(dict)
		self.ReadNetwork(network_file_l,reverse)
		self.IntegrateNetwork(inter_method='prob')
		self.ngene = len(self.g2i)

	def ReadNetwork(self, network_file_l,reverse):
		#print file_name
		self.g2i = {}
		self.i2g = {}
		self.network_i = {}
		self.gp_set = set()
		self.network_d = {}
		for file_name in network_file_l:
			self.network_i[file_name] = collections.defaultdict(dict)
			self.network_d[file_name] = collections.defaultdict(dict)
			fin = open(file_name)
			for line in fin:
				w = line.strip().split('\t')
				g1 = w[0]
				g2 = w[1]
				if reverse:
					g1,g2 = g2,g1
				#if g1 > g2:
				#	g1,g2 = g2,g1
				if g1 not in self.g2i:
					self.g2i[g1] = len(self.i2g)
					self.i2g[self.g2i[g1]] = g1
				if g2 not in self.g2i:
					self.g2i[g2] = len(self.i2g)
					self.i2g[self.g2i[g2]] = g2
				gid1 = self.g2i[g1]
				gid2 = self.g2i[g2]
				self.gp_set.add((gid1,gid2))
				self.gp_set.add((gid2,gid1))
				if len(w) > 2 and self.weighted:
					wt = float(w[2])
				else:
					wt = 1.
				self.network_i[file_name][gid1][gid2] = wt
				self.network_i[file_name][gid2][gid1] = wt
				self.network_d[file_name][g1][g2] = wt
			fin.close()

	def IntegrateNetwork(self, inter_method):
		if inter_method == 'prob':
			row_ind = []
			col_ind = []
			val = []
			ngene = len(self.g2i)
			for (i,j) in self.gp_set:
				wt = 1.
				for net in self.network_i:
					if i in self.network_i[net] and j in self.network_i[net][i]:
						wt *= (1 - self.network_i[net][i][j])
				wt = 1 - wt
				self.network[self.i2g[i]][self.i2g[j]] = wt
				row_ind.append(i)
				col_ind.append(j)
				val.append(wt)
				self.nedge += 1
			self.sparse_network = sparse.csr_matrix((val, (row_ind, col_ind)), shape=(ngene, ngene))

	def FindConnectedComponent(self,tgt, node_list):
		unvisit_nodes = set(node_list[1:])
		cur_node = set([tgt])
		pre = {}
		st = 0
		node_weight = {}
		while(len(unvisit_nodes)>0):
			new_cur_node = set()
			for g in cur_node:
				for ngh in self.network[g]:
					if ngh in pre or ngh==tgt:
						continue
					new_cur_node.add(ngh)
					pre[ngh] = g
					if ngh in unvisit_nodes:
						unvisit_nodes.remove(ngh)
			cur_node = new_cur_node
			st+=1
			#print st, len(new_cur_node),len(unvisit_nodes)
			if st>5:
				break
		node_set = set()
		node_set.add(tgt)
		edge_list = []
		for g in node_list:
			ng = g
			while ng in pre:
				node_set.add(ng)
				node_weight[ng] = 1
				edge_list.append([pre[ng],ng,self.network[pre[ng]][ng]])
				ng = pre[ng]

		return node_set, edge_list,node_weight

	def FindSteinerTree(self, tgt, node_list):
		node_list.append(tgt)
		edges =  []
		costs = []
		for g1 in self.network:
			for g2 in self.network[g1]:
				edges.append([self.g2i[g1], self.g2i[g2]])
				costs.append(10*self.network[g1][g2])

		prizes = []
		for g in range(self.ngene):
			p = int(self.i2g[g] in node_list)*100.
			prizes.append(int(self.i2g[g] in node_list)*100)
		import pcst_fast
		st_vertices, st_edges = pcst_fast.pcst_fast(edges, prizes, costs, self.g2i[tgt.upper()], 1, 'gw', 0)
		extended_genes = []
		for v in st_vertices:
			extended_genes.append(self.i2g[v].lower())
		return extended_genes

	def get_K_hop_ngh(self,root, khop=1):
		if khop==0:
			return set([root])
		if khop==-1:
			return set(self.g2i.keys())
		node = set()
		node.add(root)
		for i in range(khop):
			new_node = set()
			for g in node:
				for ngh in self.network[g]:
					new_node.add(ngh)
			for n in new_node:
				node.add(n)
		return node

	def get_network_given_genes(self, i2g):

		ppi_net = self.sparse_network.toarray()
		ngene = len(i2g)
		network = np.zeros((ngene,ngene))
		for i in range(ngene):
			gi = i2g[i]
			if gi not in self.g2i:
				continue
			mi = self.g2i[gi]
			for j in range(i,ngene):
				gj = i2g[j]
				if gj not in self.g2i:
					continue
				mj = self.g2i[gj]
				network[i,j] = ppi_net[mi, mj]
				network[j,i] = ppi_net[mi, mj]
		return network

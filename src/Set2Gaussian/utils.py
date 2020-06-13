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

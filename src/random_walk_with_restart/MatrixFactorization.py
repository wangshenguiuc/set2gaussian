import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#from tensorflow_probability import distributions as tfd
import sys
from scipy.optimize import minimize
from scipy import stats
import GPUtil
import time
import psutil


class MF:
	def __init__(self, matrix, dim=50, max_iter =100, lr = 0.01, confidence='',seed=0,lambda1=0.1):
		tf.reset_default_graph()
		tf.set_random_seed(seed)
		np.random.seed(seed)
		self.matrix = matrix
		self.n0, self.n1 = np.shape(matrix)
		self.lambda1 = lambda1
		self.dim = dim
		if len(confidence)==0:
			self.confidence = np.ones(np.shape(matrix))
		else:
			self.confidence = confidence

		self.__build()
		sys.stdout.flush()
		#self.__dataset_generator(hops, scale_terms)
		self.__build_loss()
		self.matrix_new = self.train(max_iter=max_iter,lr=lr)[0]
		sys.stdout.flush()
		# setup the validation set for easy evaluation

	def __build(self):
		w_init = tf.contrib.layers.xavier_initializer
		self.U = tf.get_variable(name='U',  dtype=tf.float32, shape=[self.n0,  self.dim ],initializer= w_init())
		self.V = tf.get_variable(name='V', dtype=tf.float32, shape=[self.n1,  self.dim],initializer= w_init())


	def __build_loss(self):
		self.p2g = tf.matmul(self.U, tf.transpose(self.V))
		self.matrix_loss =  tf.losses.mean_squared_error(tf.multiply(self.p2g,self.confidence), tf.multiply(self.matrix,self.confidence))
		self.reg_loss = self.lambda1 * ( tf.nn.l2_loss(self.U) +  tf.nn.l2_loss(self.V) )
		self.loss = self.matrix_loss + self.reg_loss

	def train(self, max_iter=100,lr=0.001,gpu_list='0'):
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
		cost_val = []
		train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

		sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
		sess.run(tf.global_variables_initializer())

		for epoch in range(max_iter):
			loss, _ = sess.run([self.loss, train_op])
			#if epoch%100==0:
			#	print epoch, loss
			#print loss
			#print epoch, loss
		self.loss = loss
		p2g = sess.run(self.p2g)
		U = sess.run(self.U)
		V = sess.run(self.V)
		tf.reset_default_graph()
		sess.close()
		return p2g, U, V

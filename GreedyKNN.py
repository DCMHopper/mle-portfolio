# Derek Hopper  2020
#
# this code is an implementation of the greedy k-nearest neighbors algorithm

import numpy as np
from sklearn.datasets import load_breast_cancer as lbc
from sklearn.metrics import roc_auc_score as auroc_score

class GreedyKNN:

	def __init__(self):
		feats = []

	def kNNpredict(self,X,y,k):
		n,m = X.shape
		y_hat = []

		for sample in range(n):
			dists = []
			for test in range(n):
				#linalg.norm defaults to euclidian distance between vectors
				euclid = np.linalg.norm(X[sample]-X[test])
				dists.append((test, euclid))
			dists.sort(key=lambda x: x[0])
			kdists = dists[:k][1]
			y_hat.append(np.mean(kdists))

		return y_hat

	def get_feature_order(self,X, y, k=5):

		n,m = X.shape
		feature_lst = []

		while len(feature_lst) < m:
			max_auroc = 0.0
			max_var = -1
			for j in range(m):
				if j in feature_lst:
					continue

				y_hat = self.kNNpredict(X[:,feature_lst + [j]], y, k)
				auroc =  auroc_score(y,y_hat) # from sklearn.metrics
				if auroc > max_auroc:
					max_auroc = auroc
					max_var = j
			feature_lst.append(max_var)

		return feature_lst

# Derek Hopper  2020
#
# this code contains two variants of a decision tree: a "quaternary" DT, which splits the data 4 ways
# instead of 2, and a "DaR" DT which extends the quaternary variant but places a ridge regressor at
# each leaf node instead of a simple mean value
#
# the quaternary modification limits the depth of the tree at the cost of flexibility. the regressor
# modification performs well in nodes with a high stdev, but fails when there are few samples in the
# node. the regressor variant could be situationally useful - the quaternary variant is just a novelty

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer as lbc
from sklearn.datasets import load_boston as boston
from sklearn.linear_model import Ridge
from math import log

class QuaternaryDecisionTree:

	def __init__(self):
		self.root = {}
		self.depth = 0

	# eval_node is its own function so we can override it in Q2
	def eval_node(self,X,y):
		return np.mean(y)

	# calculate the length/mean of two halves of a potential split
	def conditional_mean(self,X,y,var_index,val):
		lt_y,ge_y = [],[]
		for i in range(len(y)):
			if X[i,var_index] < val:
				lt_y.append(y[i])
			else:
				ge_y.append(y[i])

		return (len(lt_y),np.mean(lt_y),len(ge_y),np.mean(ge_y))

	# calculate the information gain of a potential split
	def info_gain(self,X,y,var_index,val):
		plen,pval,qlen,qval = self.conditional_mean(X,y,var_index,val)

		if pval == 0:
			pval += np.finfo(float).eps
		if qval == 0:
			qval += np.finfo(float).eps
		if pval == 1:
			pval -= np.finfo(float).eps
		if qval == 1:
			qval -= np.finfo(float).eps

		H_y_q = -qval*log(qval) - (1-qval)*log(1-qval)
		H_y_p = -pval*log(pval) - (1-pval)*log(1-pval)
		return -plen*H_y_p - qlen*H_y_q

	# select best two splits for a quat stump
	def split(self,X,y):
		var1,var2,val1,val2,max_gain1,max_gain2 = 0,0,0,0,-999,-999
		n,m = X.shape

		for test_var in range(m):
			for test_val_index in range(n):
				test_val = X[test_val_index,test_var]
				
				info_gain = self.info_gain(X,y,test_var,test_val)
				if info_gain > max_gain1:
					var1 = test_var
					val1 = test_val
					max_gain1 = info_gain

		for test_var in range(m):
			if test_var == var1:
				continue

			for test_val_index in range(n):
				test_val = X[test_val_index,test_var]
				info_gain = self.info_gain(X,y,test_var,test_val)

				if info_gain > max_gain2:
					var2 = test_var
					val2 = test_val
					max_gain2 = info_gain

#		print(var1, val1, var2, val2)
		return (var1, val1, var2, val2)

	def build_tree(self,X,y,depth):
		n,m = X.shape
		if n < 3 or depth <= 0:
#			print("layer: ", depth, " n: ", n, " val: ", np.mean(y))
			return self.eval_node(X,y)

		splitvar1, splitval1, splitvar2, splitval2 = self.split(X,y)

		# quadrant naming: ab where a is left or right on var1 and b is left
		# or right on var2, order is always ll-lr-rl-rr
		left_idx = X[:,splitvar1] < splitval1
		right_idx = X[:,splitvar1] >= splitval1

		X_l,y_l = X[left_idx,:],y[left_idx]
		X_r,y_r = X[right_idx,:],y[right_idx]

		left_left_idx = X[left_idx,splitvar2] < splitval2
		left_right_idx = X[left_idx,splitvar2] >= splitval2
		right_left_idx = X[right_idx,splitvar2] < splitval2
		right_right_idx = X[right_idx,splitvar2] >= splitval2

		X_ll,y_ll = X_l[left_left_idx,:],y_l[left_left_idx]
		X_lr,y_lr = X_l[left_right_idx,:],y_l[left_right_idx]
		X_rl,y_rl = X_r[right_left_idx,:],y_r[right_left_idx]
		X_rr,y_rr = X_r[right_right_idx,:],y_r[right_right_idx]

		return {"var1": splitvar1,
				"val1": splitval1,
				"var2": splitvar2,
				"val2": splitval2,
				"ll": self.build_tree(X_ll,y_ll,depth-1),
				"lr": self.build_tree(X_lr,y_lr,depth-1),
				"rl": self.build_tree(X_rl,y_rl,depth-1),
				"rr": self.build_tree(X_rr,y_rr,depth-1),}

	def fit(self,X,y,max_depth=2):

		self.root = self.build_tree(X,y,max_depth)
		self.depth = max_depth

		return 0

	def predict(self,X):
		pred_y = []
		for row in X:
			node = self.root
			# climb tree to leaf node
			while not isinstance(node,float):
				if row[node["var1"]] < node["val1"]:
					if row[node["var2"]] < node["val2"]:
						node = node["ll"]
					else:
						node = node["lr"]
				else:
					if row[node["var2"]] < node["val2"]:
						node = node["rl"]
					else:
						node = node["rr"]
			pred_y.append(node)

		return pred_y

class DaRDecisionTree(QuaternaryDecisionTree):

	# instead of mean(y), run a regressor and append it to the tree
	def eval_node(self,X,y):
		if len(y) < 1:
			return 0
		regressor = Ridge(alpha=1.0)
		regressor.fit(X,y)
		return regressor

	# split() maximizes info_gain, so return negative standard deviation,
	# weighted by length on each side
	def info_gain(self,X,y,var,val):

		left_idx = X[:,var] < val
		right_idx = X[:,var] >= val

		X_l,y_l = X[left_idx,:],y[left_idx]
		X_r,y_r = X[right_idx,:],y[right_idx]

		std_l,len_l = np.std(y_l),len(y_l)
		std_r,len_r = np.std(y_r),len(y_l)

		return -len_l*std_l - len_r*std_r

	# same tree climb, only now run regressor at leaf nodes
	def predict(self,X):
		pred_y = []
		for row in X:
			node = self.root
			while not isinstance(node,Ridge):
				if node == 0:
					return 0
				if row[node["var1"]] < node["val1"]:
					if row[node["var2"]] < node["val2"]:
						node = node["ll"]
					else:
						node = node["lr"]
				else:
					if row[node["var2"]] < node["val2"]:
						node = node["rl"]
					else:
						node = node["rr"]
			row = row.reshape(1,-1)
			pred_y = np.append(pred_y,node.predict(row))

		return pred_y

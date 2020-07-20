# Derek Hopper  2020
#
# this code is an implementation of the k-means clustering algorithm

import sys
import numpy as np
from sklearn.metrics import silhouette_score as s_coef

class KMeans:

	def __init__(self):
		self.parts_count = 0

	def distance(self,p1,p2):
		return np.linalg.norm(p1-p2)

	def fit(self, X, k, output):
		n,m = X.shape

		means = np.zeros((k,m))
		parts = np.zeros(n)

		# assign samples to partitions arbitrarily
		dealer = 0
		for i in range(n):
			parts[i] = dealer
			dealer = (dealer + 1) % k

		error = 0
		next_parts = np.zeros(n)
		done = False

		# loop until next means == current means
		while not done:
			error = 0

			# calculate means
			for i in range(k):
				X_part = X[parts[:] == i, :]
				means[i] = X_part.mean(axis=0)

			for i in range(n):
				best_part = 0   # which partition is closest
				best_err = 999999  # error value of closest partition
				for j in range(k):
					this_err = self.distance(means[j], X[i,:])
					if this_err < best_err:
						best_err = this_err
						best_part = j

				next_parts[i] = best_part
				error += best_err

			# have we found a solution (local minimum error)?
			done = True
			if not np.array_equal(parts,next_parts):
				done = False

			parts = next_parts
			next_parts = np.zeros(n)

		print(X)
		print(parts)
		silhouette = s_coef(X,parts,metric='l2')

		with open(output, 'w') as out:
			for i in range(n):
				out.write( str(parts[i]) + ", " + ", ".join(str(x) for x in X[i,:]) + "\n")
			out.write("SSE: " + str(error) + ", Silhouette Score: " + str(silhouette))

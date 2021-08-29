from decision_tree import DecisionTree as DT
import numpy as np
import matplotlib.pyplot as plt

# Fit knn on sample of size 30
# Consider plot the way future samples get labelled for varying k

plt.style.use('ggplot')

N = 1000
K = 6
true_labels = np.random.randint(0, K, N)
true_means = np.random.rand(K, 2)*10
covs = np.empty((K, 2, 2))
for i in range(K):
	cov_ = np.random.rand(2)
	covs[i] = np.outer(cov_, cov_)
	
X = np.empty((N, 2))
for i in range(N):
	X[i] = np.random.multivariate_normal(true_means[true_labels[i]], np.identity(2))

plt.subplot(5, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=true_labels)
plt.title('Training sample')
plt.xticks([])
plt.yticks([])

dt = DT(reg_clas=True)
dt.fit(X, true_labels, verbose=True)

N = 10000
true_labels = np.random.randint(0, K, N)

X = np.empty((N, 2))
for i in range(N):
	X[i] = np.random.multivariate_normal(true_means[true_labels[i]], np.identity(2))

plt.subplot(5, 1, 2)
plt.scatter(X[:, 0], X[:, 1], c=true_labels)
plt.title('True large sample')
plt.xticks([])
plt.yticks([])

classified_labels1 = dt.predict(X)

plt.subplot(5, 1, 3)
plt.scatter(X[:, 0], X[:, 1], c=classified_labels1)
plt.title('Classified decision tree')
plt.xticks([])
plt.yticks([])

# Validation set:
V = 30
y_ = np.random.randint(0, K, V)
X_ = np.empty((V, 2))
for i in range(V):
	X_[i] = np.random.multivariate_normal(true_means[y_[i]], np.identity(2))

plt.subplot(5, 1, 4)
plt.scatter(X_[:, 0], X_[:, 1], c=y_)
plt.title('Validation set')
plt.xticks([])
plt.yticks([])

dt.prune(X_, y_, verbose=True)
classified_labels2 = dt.predict(X)

plt.subplot(5, 1, 5)
plt.scatter(X[:, 0], X[:, 1], c=classified_labels2)
plt.title('Pruned decision tree )')
plt.xticks([])
plt.yticks([])

plt.show()

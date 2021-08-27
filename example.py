from decision_tree import DecisionTree as DT
import numpy as np
import matplotlib.pyplot as plt

# Fit knn on sample of size 30
# Consider plot the way future samples get labelled for varying k

N = 50
K = 4
true_labels = np.random.randint(0, K, N)
true_means = np.random.rand(K, 2)*10
covs = np.empty((K, 2, 2))
for i in range(K):
	cov_ = np.random.rand(2)
	covs[i] = np.outer(cov_, cov_)
	
X = np.empty((N, 2))
for i in range(N):
	X[i] = np.random.multivariate_normal(true_means[true_labels[i]], np.identity(2))

plt.subplot(4, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=true_labels)
plt.title('Training sample')
plt.xticks([])
plt.yticks([])

dt = DT(reg_clas=True)
dt.fit(X, true_labels)

dtR = DT(reg_clas=True, l=1)
dtR.fit(X, true_labels)

N = 10000
true_labels = np.random.randint(0, K, N)

X = np.empty((N, 2))
for i in range(N):
	X[i] = np.random.multivariate_normal(true_means[true_labels[i]], np.identity(2))

plt.subplot(4, 1, 2)
plt.scatter(X[:, 0], X[:, 1], c=true_labels)
plt.title('True large sample')
plt.xticks([])
plt.yticks([])

classified_labels1 = dt.predict(X)

plt.subplot(4, 1, 3)
plt.scatter(X[:, 0], X[:, 1], c=classified_labels1)
plt.title('Classified decision tree')
plt.xticks([])
plt.yticks([])

classified_labels2 = dtR.predict(X)

plt.subplot(4, 1, 4)
plt.scatter(X[:, 0], X[:, 1], c=classified_labels2)
plt.title('Classified decision tree (regularised) TODO')
plt.xticks([])
plt.yticks([])

plt.show()

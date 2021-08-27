import numpy as np
import matplotlib.pyplot as plt

class node:

    """
    parent:         parent node of new node.
    D:              list of indices of observations node must contain.      
    observations:   tupe of two np arrays, X and y, X: (n, p), y: (n,)
    """
    def __init__(self, parent, D=None, cost_f=None, observations=None):
        if parent is None:
            if observations is None:
                print('If no parent specified then requires observations (X, y) to be specified.')
                return
            if cost_f is None:
                print('If no parent specified then requires cost function to be specified.')
            self.cost_f = cost_f
            self.X, self.y = observations 
            self.cost_f = cost_f
            if D is None:
                self.D = [i for i in range(self.X.shape[0])]
        else:
            self.X = parent.X
            self.y = parent.y
            self.cost_f = parent.cost_f
            if D is not None:
                self.D = D
            else:
                self.D = parent.D
        self.l = None
        self.r = None
        self.p = parent
        self.split = None
        self.cost = self.cost_f(self.y[self.D])

    def split_node(self, split):
        self.split = split
        left_d, right_d = self._split_data()
        self.l = node(self, D=left_d)
        self.r = node(self, D=right_d)

    def _split_data(self, split=None):
        if split is None:
            split = self.split
        l = []
        r = []
        for d in self.D:
            i, v = split
            if self.X[d, i] < v:
                l.append(d)
            else:
                r.append(d)
        return l, r

    def _split_cost(self, split=None):
        if split is None:
            split = self.split
        l, r = self._split_data(split)
        l = self.y[l]
        r = self.y[r]
        n = len(l)
        m = len(r)
        total_cost = (n*self.cost_f(l) + m*self.cost_f(r))/(n + m)
        return total_cost

    def find_best_split(self, return_cost=False):
        lowest_cost = float('inf')
        best_split = None
        for p in range(self.X.shape[1]):
            for val in np.unique(self.X[self.D, p])[1:]:
                split = (p, val)
                cost = self._split_cost(split)
                if cost < lowest_cost:
                    best_split = split
                    lowest_cost = cost
        if return_cost:
            return best_split, lowest_cost
        else:
            return best_split

    def __str__(self):
        return str(self.X[self.D])

class DecisionTree:
    
    def __init__(self, reg_clas=False, l=0, MAX_SPLITS=1000):
        self.MAX_SPLITS = MAX_SPLITS
        self.l = 0
        self.reg_clas = reg_clas
        if reg_clas:
            self.cost_f = DecisionTree.entropy
        else:
            self.cost_f = DecisionTree.mse

    def _stop_split_cond(self, leaf):
        if self.reg_clas:
            return np.abs(leaf.cost) == 0
        else:
            return len(leaf.D) < 4

    def __str__(self):
        l = []
        i = 0
        DecisionTree._traverse(self.root, l, i)
        return ' -- '.join(l)

    def _traverse(node, l, i):
        l.append(str(i))
        i += 1
        l.append(str(node.split))
        if node.l is not None and node.l.split is not None:
            l.append('l')
            DecisionTree._traverse(node.l, l, i)
        if node.r is not None and node.r.split is not None:
            l.append('r')
            DecisionTree._traverse(node.r, l, i)

    def new_split(self, leaf_index, split):
        p = self.leaves[leaf_index]
        p.split_node(split)
        self.leaves.pop(leaf_index)
        self.leaves.insert(leaf_index, p.r)
        self.leaves.insert(leaf_index, p.l)

    def prune(self):
        print('TODO')

    def fit(self, X, y):
        self.N, self.d = X.shape
        if y.shape != (self.N,):
            print('Incorrect dimensions training data')
            return
        self.root = node(parent=None, cost_f=self.cost_f, D=None, observations=(X, y))
        self.leaves = [self.root]
        n_splits = 0
        i = 0
        while i < len(self.leaves):
            if self._stop_split_cond(self.leaves[i]):
                i = i + 1
                continue
            split, cost = self.leaves[i].find_best_split(return_cost=True)
            self.new_split(i, split)
            n_splits += 1
            if n_splits >= self.MAX_SPLITS:
                break
        print(f'Number of splits: {n_splits}')
    
    def predict(self, X):
        if self.reg_clas:
            yhat = np.empty(X.shape[0], dtype=int)
        else:
            yhat = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            x = X[i]
            yhat[i] = self._predict(x)
        return yhat

    def _predict(self, x):
        cur = self.root
        while cur.split != None:
            p, v = cur.split
            if x[p] < v:
                cur = cur.l
            else:
                cur = cur.r
        if self.reg_clas:
            return DecisionTree._mode(cur.y[cur.D])
        else:
            return DecisionTree._mean(cur.y[cur.D])

    def _mean(y):
	    return np.mean(y)

    def _mode(y):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]

    def mse(y):
        if len(y) == 0:
            return 0
        ybar = np.mean(y)
        return np.mean((y - ybar)**2)
	
    def entropy(y):
        if len(y) == 0:
            return 0
        unique, counts = np.unique(y, return_counts=True)
        P = counts / np.sum(counts)
        return -np.sum(P * np.log2(P))

if __name__ == '__main__':
    dt = DecisionTree(reg_clas=True)
    X = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    y = np.array([0, 1, 0, 1])
    dt.fit(X, y)
    print(dt)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    grid = (np.random.rand(10000, 2) - .5) * 10
    grid_hat = dt.predict(grid)
    plt.scatter(grid[:, 0], grid[:, 1], c=grid_hat)
    plt.show()

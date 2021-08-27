import numpy as np
import matplotlib.pyplot as plt

class node:

    """
    parent:         parent node of new node.
    D:              list of indices of observations node must contain.      
    observations:   tupe of two np arrays, X and y, X: (n, p), y: (n,)
    """
    def __init__(self, parent, D=None, observations=None):
        if observations is None:
            if parent is None:
                print('If no parent specified then requires observations (X, y) to be specified.')
                return
            self.X = parent.X
            parent.y = parent.y
        else:
            self.X, self.y = observations 
        if D is None:
            self.D = [i for i in range(self.X.shape[0])]
        else:
            self.D = D
        self.l = None
        self.r = None
        self.p = parent
        self.split = None

    def split_node(self, predictor_index, split_val):
        self.split = (predictor_index, split_val)
        left_d, right_d = self._split_data()
        self.l = node(self, left_d)
        self.r = node(self, right_d)

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

    def _split_cost(self, cost_f, split=None):
        if split is None:
            split = self.split
        l, r = self._split_data(split)
        l = self.y[l]
        r = self.y[r]
        n = len(l)
        m = len(r)
        total_cost = (n*cost_f(l) + m*cost_f(r))/(n + m)
        return total_cost

    def find_best_split(self, cost_f):
        lowest_cost = float('inf')
        best_split = None
        for p in range(self.X.shape[1]):
            self.D.sort(key=lambda d: self.X[d, p])
            prev_val = None
            for val in self.X[self.D, p]:
                if val == prev_val:
                    continue
                prev_val = val
                split = (p, val)
                cost = self._split_cost(cost_f, split)
                print(f'test: {(p, val)}')
                print(f'cost: {cost}')
                if cost < lowest_cost:
                    best_split = split
                    lowest_cost = cost
        return best_split

    def __str__(self):
        return str(self.X[self.D])

class DecisionTree:
    
    def __init__(self, l=0):
        self.l = 0

    def __str__(self):
        return '\n'.join([str(n) for n in self.leaves])

    def new_split(self, leaf_index, predictor_index, split_val):
        p = self.leaves[leaf_index]
        p.split_node(predictor_index, split_val)
        self.leaves.insert(leaf_index, p.r)
        self.leaves.insert(leaf_index, p.l)
        self.leaves.pop(leaf_index + 2)

    def prune(self):
        print('TODO')

    def fit(self, X, y, reg_clas=False):
        self.N, self.d = X.shape
        if y.shape != (self.N,):
            print('Incorrect dimensions training data')
            return
        self.root = node(parent=None, D=None, observations=(X, y))
        self.leaves = [self.root]

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
    dt = DecisionTree()
    X = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    y = np.array([0, 1, 1, 0])
    dt.fit(X, y)
    print(dt.leaves[0].find_best_split(DecisionTree.entropy))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

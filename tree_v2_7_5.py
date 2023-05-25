from __future__ import division
from collections import namedtuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize
from collections import Counter

def find_best_split(feature_vector, target_vector, task="classification"):
    def dispersion(y):
        return np.sum((y - np.mean(y)) ** 2)

    def gini(y):
        p1 = np.sum(y)
        p0 = 1. - p1 / y.shape[0]
        p1 /= y.shape[0]
        return 1 - p0**2 - p1**2

    def impurity(task, X, y, threshold):
        split = np.array(X <= threshold)
        l = y[split]
        r = y[~split]
        if task == "regression":
            l_criteria = dispersion(l)
            r_criteria = dispersion(r)
        else:
            l_criteria = gini(l)
            r_criteria = gini(r)
        return - l.shape[0] / y.shape[0] * l_criteria - r.shape[0] / y.shape[0] * r_criteria

    index = np.argsort(feature_vector)
    feature_vector = feature_vector[index]
    target_vector = target_vector[index]
    thresholds, impurities, threshold_best, impurity_best = [], [], -1, -np.inf
    for i in range(feature_vector.shape[0] - 1):
        cur_threshold = (feature_vector[i] + feature_vector[i+1]) / 2
        if (target_vector[feature_vector <= cur_threshold].shape[0] == 0) or (target_vector[feature_vector > cur_threshold].shape[0] == 0):
            continue
        cur_impurity = impurity(task, feature_vector, target_vector, cur_threshold)
        thresholds.append(cur_threshold)
        impurities.append(cur_impurity)
        if cur_impurity > impurity_best:
            threshold_best, impurity_best = cur_threshold, cur_impurity

    return thresholds, impurities, threshold_best, impurity_best

class DecisionTree(object):
    def __init__(self, max_depth=None, min_samples_split=None, min_samples_leaf=None, task="classification"):
        self._tree = {}
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self.task = task

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_vector = sub_X[:, feature]
            _, _, threshold, gini = find_best_split(feature_vector, sub_y, self.task)

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                threshold_best = threshold

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        node["left_child"], node["right_child"] = {}, {}

        split = np.array(sub_X[:, feature_best] <= threshold_best)
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        if x[int(node["feature_split"])] <= float(node["threshold"]):
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

data = np.genfromtxt("sdss_redshift.csv", delimiter=',', skip_header=1)
x = data[:, 0:5]
y = data[:, 5]

# Split the data into train and test sets
np.random.seed(np.random.randint(0, np.random.randint(50, 1000)))
indices = np.random.permutation(len(x))
train_size = int(0.9 * len(x))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
X_train, X_test = x[train_indices], x[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

k = 1
print x.shape
forest = []
while k < 3:
    k1 = 0
    print y_train
    garden = []
    while k1 < int(10**(1/k)):
        np.random.seed(np.random.randint(0, 20))
        train_indices = np.random.permutation(len(X_train))
        train_size = int(0.5 * len(X_train))
        train_split_indices = train_indices[:train_size]
        X_train_split, y_train_split = X_train[train_split_indices], y_train[train_split_indices]
        tree = DecisionTree(max_depth=None, min_samples_split=None, min_samples_leaf=None, task="classification")
        tree.fit(X_train_split, y_train_split)
        print "Acc =", np.mean(np.abs(y_train - tree.predict(X_train))) / np.mean(np.abs(y_train))
        k1 += 1
        garden.append(tree)
    y_train = y_train - np.mean([tree.predict(X_train) for tree in garden], axis=0)
    forest.append(garden)
    k += 1


y_pred = np.sum([np.mean([tree.predict(X_test.values) for tree in garden], axis=0) for garden in forest], axis=0)

plt.title("Prediction")
plt.plot(y_test, y_pred, '.', label='Data', alpha=0.3)
plt.grid(True)
plt.legend()
plt.savefig("redshift.png")

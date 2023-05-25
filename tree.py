from    collections import namedtuple
import  matplotlib.pyplot as plt
from    scipy import optimize
import  numpy as np
from    sklearn.model_selection import train_test_split
import  pandas as pd
from typing import Dict, List, Tuple, Union
from collections import Counter

def find_best_split(
    feature_vector: np.ndarray,
    target_vector: np.ndarray,
    task: str = "classification"
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
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

class DecisionTree:
    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = None,
        min_samples_leaf: int = None,
        task: str = "classification"
    ) -> None:
        self._tree = {}
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self.task = task

    def _fit_node(
        self,
        sub_X: np.ndarray,
        sub_y: np.ndarray,
        node: dict
    ) -> None:
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

    def _predict_node(self, x: np.ndarray, node: dict) -> int:
        if node["type"] == "terminal":
            return node["class"]
        if x[int(node["feature_split"])] <= float(node["threshold"]):
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._fit_node(X, y, self._tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

data = pd.read_csv("sdss_redshift.csv")
x = data[['u', 'g', 'r', 'i', 'z']]
y = data['redshift']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=np.random.randint(0, np.random.randint(50, 1000)))

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

k = 1
print(x.shape)
forest = []
while k < 3:
    k1 = 0
    print(y_train)
    garden = []
    while k1 < 10**(1/k):
        X_train_split, _, y_train_split, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=np.random.randint(0, 20))
        tree = DecisionTree(max_depth=None, min_samples_split=None, min_samples_leaf=None, task="classification")
        tree.fit(X_train_split.to_numpy(), y_train_split.to_numpy())
        print("Acc =", np.mean(np.abs(y_train - tree.predict(X_train.to_numpy()))) / np.mean(np.abs(y_train)))
        k1 += 1
        garden.append(tree)
    y_train = y_train - np.mean([tree.predict(X_train.to_numpy()) for tree in garden], axis=0)
    forest.append(garden)
    k += 1

y_pred = np.sum([np.mean([tree.predict(X_test.to_numpy()) for tree in garden], axis=0) for garden in forest], axis=0)

plt.title("Prediction")
plt.plot(y_test, y_pred, '.', label='Data', alpha=0.3)
plt.grid(True)
plt.legend()
plt.savefig("redshift.png")

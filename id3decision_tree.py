import numpy as np


class Node:
    def __init__(self, feature, children, label):
        self.feature = feature
        self.children = children
        self.label = label


def id3(X, y, features):
    if len(np.unique(y)) == 1:
        return Node(None, {}, y[0])

    if len(features) == 0:
        return Node(None, {}, np.bincount(y).argmax())

    best_feature = max(features, key=lambda f: information_gain(X, y, f))
    remaining_features = [f for f in features if f != best_feature]
    children = {}

    for value in np.unique(X[:, best_feature]):
        idx = X[:, best_feature] == value
        child = id3(X[idx], y[idx], remaining_features)
        children[value] = child

    return Node(best_feature, children, None)


def information_gain(X, y, feature):
    entropy_before = entropy(y)
    entropy_after = 0
    for value in np.unique(X[:, feature]):
        idx = X[:, feature] == value
        entropy_after += entropy(y[idx]) * len(y[idx]) / len(y)
    return entropy_before - entropy_after


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -sum(probabilities * np.log2(probabilities))


# data
data = np.array(
    [
        ["Sunny", "Hot", "High", "Weak", "No"],
        ["Sunny", "Hot", "High", "Strong", "No"],
        ["Overcast", "Hot", "High", "Weak", "Yes"],
        ["Rain", "Mild", "High", "Weak", "Yes"],
        ["Rain", "Cool", "Normal", "Weak", "Yes"],
        ["Rain", "Cool", "Normal", "Strong", "No"],
        ["Overcast", "Cool", "Normal", "Strong", "Yes"],
        ["Sunny", "Mild", "High", "Weak", "No"],
        ["Sunny", "Cool", "Normal", "Weak", "Yes"],
        ["Rain", "Mild", "Normal", "Weak", "Yes"],
        ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ]
)

X, y = data[:, :-1], data[:, -1]
features = [i for i in range(X.shape[1])]
tree = id3(X, y, features)

# function to print the decision tree
def print_tree(node, prefix=""):
    if node.label is not None:
        print(prefix + "->", node.label)
    else:
        print(prefix + "->", node.feature)
        for value, child_node in node.children.items():
            print(prefix + " |")
            print_tree(child_node, prefix + " " + str(value))


# print the decision tree
print_tree(tree)

#!/usr/bin/env python3
"""
A function to display the textual structure of a trained
decision tree classifier.

Arguments:
clf: A trained DecisionTreeClassifier instance from Scikit-learn
feature_names: A list of the input feature names
class_names: A list of the target class names
Returns:

None. The function prints a readable text representation of the
decision tree structure.
"""


def draw(clf, feature_names, class_names):
    """
    A function to display the textual structure of a trained
    decision tree classifier
    """
    tree = clf.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold.round(2)
    values = tree.value

    def print_node(curr, depth):
        """ Recursively print the tree structure """

        print("|   " * (depth - 1) + "|--- ", end="")
        if children_left[curr] != children_right[curr]:
            print(f"{feature_names[feature[curr]]} <= {threshold[curr]}")
            print_node(children_left[curr], depth + 1)
            print("|   " * (depth - 1) + "|--- ", end="")
            print(f"{feature_names[feature[curr]]} >  {threshold[curr]}")
            print_node(children_right[curr], depth + 1)
        else:
            print(f"class: {class_names[values[curr].argmax()]}")

    print_node(0, 1)

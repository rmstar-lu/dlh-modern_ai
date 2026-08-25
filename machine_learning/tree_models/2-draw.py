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
from sklearn import tree


def draw(clf, feature_names, class_names):
    """
    A function to display the textual structure of a trained
    decision tree classifier
    """
    print(tree.export_text(clf,
                           feature_names=feature_names,
                           class_names=class_names))

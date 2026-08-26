#!/usr/bin/env python3
"""
A function that selects the best pruning value ccp_alpha for a set of trained
decision trees.

This function first identifies the model(s) that achieve the highest test
accuracy.
If multiple models share this same test accuracy, it selects the one with the
smallest difference between training and test accuracy to favor better
generalization.
In the event of a further tie, the model associated with the largest ccp_alpha
is chosen to promote a simpler, more regularized tree.

Arguments:
clfs: List of trained DecisionTreeClassifier instances, each trained with a
different ccp_alpha.
train_scores: List of training accuracy scores corresponding to each
classifier in clfs.
test_scores: List of test accuracy scores corresponding to each classifier in
clfs as well.
ccp_alphas: List or array of ccp_alpha values used to train the classifiers.

Returns:
best_alpha: The most appropriate ccp_alpha value based on test accuracy and
generalization.
best_clf: The trained classifier associated with the best alpha.
"""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """
    Select the best pruning value ccp_alpha for a set of trained
    decision trees.
    """
    best_acc = max(test_scores)
    indices = [i for i, acc in enumerate(test_scores) if acc == best_acc]
    if len(indices) == 1:
        return (ccp_alphas[indices[0]], clfs[indices[0]])

    diff = [abs(train_scores[i] - test_scores[i]) for i in indices]
    min_diff = min(diff)
    indices = [ix for i, ix in enumerate(indices) if diff[i] == min_diff]
    if len(indices) == 1:
        return (ccp_alphas[indices[0]], clfs[indices[0]])

    largest_alpha = max((ccp_alphas[i] for i in indices))
    i = next((i for i in indices if ccp_alphas[i] == largest_alpha))
    return (ccp_alphas[i], clfs[i])

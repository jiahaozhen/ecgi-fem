from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class CMLClassifier(BaseEstimator, ClassifierMixin):
    """
    Collective Multi-Label Classifier (CML)
    This is a simplified implementation inspired by the CML concept.
    The idea is to iteratively refine the predictions by using the predictions of other labels
    as additional features. This is similar to a stacked approach or a self-learning approach
    where relationships between labels are learned collectively.

    Algorithm:
    1. Train independent binary classifiers for each label (Binary Relevance).
    2. Obtain initial predictions (probabilities) for all training data.
    3. Construct an extended feature set: X_new = [X, Predictions]
    4. Train a second layer of classifiers on X_new to predict y.

    This captures label correlations because the second layer sees the predicted probabilities
    of all other labels.
    """

    def __init__(self, base_estimator=None, max_iterations=1):
        self.base_estimator = base_estimator
        self.max_iterations = max_iterations

    def fit(self, X, y):
        self.n_labels_ = y.shape[1]

        if self.base_estimator is None:
            self.base_estimator = RandomForestClassifier(
                n_estimators=100, random_state=42
            )

        # Layer 1: Binary Relevance
        self.classifiers_l1_ = []
        for i in range(self.n_labels_):
            clf = clone(self.base_estimator)
            clf.fit(X, y[:, i])
            self.classifiers_l1_.append(clf)

        # Get initial predictions (probabilities) on training data
        # Note: Ideally we should use cross-validation here to avoid overfitting,
        # but for a basic implementation we use direct prediction.
        y_pred_proba = np.zeros((X.shape[0], self.n_labels_))
        for i, clf in enumerate(self.classifiers_l1_):
            y_pred_proba[:, i] = clf.predict_proba(X)[:, 1]

        # Layer 2: Collective step
        # Augment X with predicted probabilities
        X_augmented = np.hstack([X, y_pred_proba])

        self.classifiers_l2_ = []
        for i in range(self.n_labels_):
            clf = clone(self.base_estimator)
            clf.fit(X_augmented, y[:, i])
            self.classifiers_l2_.append(clf)

        return self

    def predict_proba(self, X):
        # Step 1: Initial predictions
        y_pred_proba_l1 = np.zeros((X.shape[0], self.n_labels_))
        for i, clf in enumerate(self.classifiers_l1_):
            y_pred_proba_l1[:, i] = clf.predict_proba(X)[:, 1]

        # Step 2: Augmented features
        X_augmented = np.hstack([X, y_pred_proba_l1])

        # Step 3: Refined predictions
        y_pred_proba_l2 = np.zeros((X.shape[0], self.n_labels_))
        for i, clf in enumerate(self.classifiers_l2_):
            y_pred_proba_l2[:, i] = clf.predict_proba(X_augmented)[:, 1]

        return y_pred_proba_l2


def multilabel_cml_classifier(base_classifier=None):
    return CMLClassifier(base_estimator=base_classifier)

import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset
import scipy.sparse as sp


class RandomKLabelsets(BaseEstimator, ClassifierMixin):
    """
    Random k-Labelsets (RAKEL) implementation.
    This is an ensemble method that trains multiple Label Powerset classifiers
    on random subsets of labels.

    This implementation corresponds to the RAKEL-d (disjoint) or RAKEL-o (overlapping)
    depending on how subsets are generated, but generally RAKEL refers to the overlapping version
    where we take 'model_count' random subsets of size 'labelset_size'.
    """

    def __init__(
        self, base_classifier=None, labelset_size=3, model_count=10, random_state=None
    ):
        self.base_classifier = base_classifier
        self.labelset_size = labelset_size
        self.model_count = model_count
        self.random_state = random_state

    def fit(self, X, y):
        # y: (n_samples, n_labels)
        self.n_labels_ = y.shape[1]

        if self.base_classifier is None:
            self.base_classifier = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            )

        np.random.seed(self.random_state)
        random.seed(self.random_state)

        self.models_ = []
        self.label_subsets_ = []
        self.label_counts_ = np.zeros(self.n_labels_)

        all_labels = np.arange(self.n_labels_)

        for _ in range(self.model_count):
            # 1. Select random label subset of size k
            # ensuring k <= n_labels
            k = min(self.labelset_size, self.n_labels_)
            subset_indices = np.random.choice(all_labels, size=k, replace=False)

            # Store subset
            self.label_subsets_.append(subset_indices)
            self.label_counts_[subset_indices] += 1

            # 2. Prepare data for this subset
            # y_subset: columns corresponding to the subset
            y_subset = y[:, subset_indices]

            # 3. Train Label Powerset on this subset
            # We use skmultilearn's LabelPowerset to handle the mapping of label sets to classes
            classifier = LabelPowerset(
                classifier=clone(self.base_classifier), require_dense=[False, True]
            )
            classifier.fit(X, y_subset)

            self.models_.append(classifier)

        return self

    def predict_proba(self, X):
        # Returns probability (voting ratio) for each label
        n_samples = X.shape[0]
        vote_sum = np.zeros((n_samples, self.n_labels_))
        vote_counts = np.zeros(self.n_labels_)
        # Ideally vote_counts should match self.label_counts_ but let"s recalculate to be safe

        for i, model in enumerate(self.models_):
            subset_indices = self.label_subsets_[i]

            # Predict for the subset. LabelPowerset.predict returns (n_samples, n_subset_labels)
            # usually as a sparse matrix.
            pred_subset = model.predict(X)

            # Ensure it's dense array for summation
            if sp.issparse(pred_subset):
                pred_subset = pred_subset.toarray()

            vote_sum[:, subset_indices] += pred_subset
            vote_counts[subset_indices] += 1

        # Avoid division by zero for labels that were never selected (though unlikely if m is large enough)
        # Masks for valid counts
        valid_mask = vote_counts > 0

        proba = np.zeros((n_samples, self.n_labels_))
        proba[:, valid_mask] = vote_sum[:, valid_mask] / vote_counts[valid_mask]

        return proba


def random_k_labelsets_classifier(
    base_classifier=None, labelset_size=3, model_count=10
):
    return RandomKLabelsets(
        base_classifier=base_classifier,
        labelset_size=labelset_size,
        model_count=model_count,
        random_state=42,
    )

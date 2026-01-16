import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier


class CalibratedLabelRanking(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        # y is (n_samples, n_labels)
        self.n_labels_ = y.shape[1]

        # Use RandomForestClassifier as default
        if self.base_estimator is None:
            self.base_estimator = RandomForestClassifier(
                n_estimators=100, random_state=42
            )

        # Pairwise classifiers (i, j) where i < j
        self.pairwise_classifiers_ = {}
        # Calibration classifiers (i, virtual_label)
        self.calibration_classifiers_ = []

        # Train calibration classifiers (Binary Relevance) i vs Virtual
        # Virtual label acts as "not relevant".
        # So we check if label i is Relevant (1) or Not Relevant (0)
        # This is exactly what Binary Relevance does.
        # If model predicts 1, it means i > Virtual.
        # If model predicts 0, it means Virtual > i.
        for i in range(self.n_labels_):
            clf = clone(self.base_estimator)
            # Handle case where a label is always 0 or always 1
            if len(np.unique(y[:, i])) < 2:
                # Store a dummy that always predicts the unique value
                # Or just Handle in predict.
                # We will store the constant value in a trivial object or wrapper
                # For simplicity, we assume sklearn deals with it (it usually throws or warns)
                # Actually sklearn handles single class fitting if we are careful? No it needs 2 classes.
                # Let's handle it manually.
                pass

            # If standard fit fails due to 1 class, we need to handle it.
            try:
                clf.fit(X, y[:, i])
                self.calibration_classifiers_.append(clf)
            except ValueError:
                # Assuming constant label
                val = y[0, i]
                self.calibration_classifiers_.append(ConstantClassifier(val))

        # Train pairwise classifiers
        for i in range(self.n_labels_):
            for j in range(i + 1, self.n_labels_):
                # Mask: only samples where labels differ
                mask = y[:, i] != y[:, j]
                count_diff = np.sum(mask)

                if count_diff == 0:
                    # Labels i and j are always same (both 0 or both 1)
                    # No preference info.
                    self.pairwise_classifiers_[(i, j)] = None
                    continue

                # If all differences are in one direction?
                # e.g. i=1,j=0 exists, but i=0,j=1 does not.
                # Then y_i_subset will be all 1.
                # sklearn handles 1-class training if we provide classes, or we catch it.

                X_subset = X[mask]
                y_i_subset = y[mask, i]  # 1 if i>j, 0 if j>i

                clf = clone(self.base_estimator)
                try:
                    clf.fit(X_subset, y_i_subset)
                    self.pairwise_classifiers_[(i, j)] = clf
                except ValueError:
                    # One class present
                    val = y_i_subset[0]
                    self.pairwise_classifiers_[(i, j)] = ConstantClassifier(val)

        return self

    def decision_function(self, X):
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_labels_))
        virtual_votes = np.zeros(n_samples)

        # Collect votes from calibration classifiers
        for i, clf in enumerate(self.calibration_classifiers_):
            preds = clf.predict(X)
            votes[:, i] += preds
            virtual_votes += 1 - preds

        # Collect votes from pairwise classifiers
        for (i, j), clf in self.pairwise_classifiers_.items():
            if clf is None:
                continue

            # Pred 1: i > j. Pred 0: j > i.
            preds = clf.predict(X)
            votes[:, i] += preds
            votes[:, j] += 1 - preds

        # Return score relative to virtual label
        return votes - virtual_votes[:, None]

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores > 0).astype(int)


class ConstantClassifier:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(X.shape[0], self.value)


def calibrated_label_ranking_classifier(base_classifier=None):
    return CalibratedLabelRanking(base_estimator=base_classifier)

from sklearn.tree import DecisionTreeClassifier


def multilabel_decision_tree_classifier():
    """
    Multi-Label Decision Tree.
    Scikit-learn's DecisionTreeClassifier supports multi-label classification natively.
    """
    classifier = DecisionTreeClassifier(random_state=42)
    return classifier

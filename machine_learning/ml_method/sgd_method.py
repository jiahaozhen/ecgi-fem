from sklearn.linear_model import SGDClassifier
from utils.machine_learning_tools import (
    load_dataset,
    exclude_classes,
    split_dataset,
    evaluate_model,
)


def sgd_classifier(X_train, y_train):
    clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
    clf.fit(X_train, y_train)
    return clf


if __name__ == '__main__':
    data_dir = (
        'machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/'
    )
    X, y = load_dataset(data_dir)
    X, y = exclude_classes(X, y, exclude_labels=[-1])
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    clf = sgd_classifier(X_train, y_train)
    evaluate_model(clf, X_test, y_test)

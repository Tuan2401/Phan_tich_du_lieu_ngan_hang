import numpy as np
from sklearn.model_selection import train_test_split


def create_pseudo_labels(model, X, y):

    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    model.fit(X_labeled, y_labeled)

    pseudo_labels = model.predict(X_unlabeled)

    X_new = np.concatenate([X_labeled, X_unlabeled])
    y_new = np.concatenate([y_labeled, pseudo_labels])

    model.fit(X_new, y_new)

    return model
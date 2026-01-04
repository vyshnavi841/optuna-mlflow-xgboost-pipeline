import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def load_and_split_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load a regression dataset and split it into train and test sets.
    This implementation is fully offline-safe.
    """

    # Load offline dataset (no internet required)
    data = load_diabetes()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

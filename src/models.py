import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ThresholdRuleClassifier(BaseEstimator, ClassifierMixin):
    """Baseline rule-based classifier: predicts 1 if a column crosses a threshold."""

    def __init__(self, column, threshold, operator=">"):
        """
        Args:
            column: name of the column to threshold.
            threshold: value to compare the column against.
            operator: ">" or ">=".
        """
        self.column = column
        self.threshold = threshold
        self.operator = operator

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        if self.operator == ">":
            mask = X[self.column] > self.threshold
        else:
            mask = X[self.column] >= self.threshold
        return mask.astype(int).to_numpy()
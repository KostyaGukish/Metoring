import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import copy


class RecursivePipeline(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_batch(self, group):
        previous_date = self.pipeline.named_steps["transformer"].data["date"].max()
        current_date = group.date.values[0]
        if current_date <= previous_date:
            raise ValueError(
                f"current_date {current_date} must be greater than previous_date {previous_date}"
            )

        pred = self.pipeline.predict(group)
        self.pipeline.named_steps["transformer"].update(group, pred)
        return pred

    def predict(self, X=None):
        X = copy.deepcopy(X)
        X_cols = X.columns
        predictions = []

        for name, group in X.sort_values("date").groupby(by=["date"], group_keys=False)[X_cols]:
            predictions.append(self.predict_batch(group))

        predictions = np.concatenate(predictions)
        predictions[predictions < 0] = 0

        X = X.sort_values("date")
        X["pred"] = predictions
        X = X.sort_index()

        return X["pred"].values

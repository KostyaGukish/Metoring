import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from tqdm import tqdm
tqdm.pandas()


class RecursivePipeline(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_batch(self, group):
        pred = self.pipeline.predict(group)
        self.pipeline.named_steps["transformer"].update(group, pred)
        return pred

    def predict(self, X=None):
        X_cols = X.columns
        predictions = (
            X.sort_values("date")
            .groupby(by=["date"], group_keys=False)[X_cols]
            .progress_apply(
                lambda group: self.predict_batch(group), include_groups=False
            )
        )

        return np.concatenate(predictions.to_numpy())
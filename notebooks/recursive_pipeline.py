import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import copy


class RecursivePipeline(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline):
        self.store_info = pd.read_csv("data/raw/air_store_info.csv")
        self.store_info = self.store_info.rename(
            columns={
                "air_store_id": "store_id",
                "air_genre_name": "genre_name",
                "air_area_name": "area_name",
            }
        )   

        self.date_info = pd.read_csv("data/raw/date_info.csv")
        self.date_info = self.date_info.rename(columns={"calendar_date": "date"})
        self.date_info["date"] = self.date_info["date"].astype("string")

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
        # X["date"] = pd.to_datetime(X["date"])

        ids = X['store_id'].unique()
        date_range = pd.date_range(start=X['date'].min(), end=X['date'].max(), freq='D')
        id_date_combinations = pd.MultiIndex.from_product([ids, date_range], names=['store_id', 'date'])
        full_df = pd.DataFrame(index=id_date_combinations).reset_index()

        self.date_info["date"] = pd.to_datetime(self.date_info["date"])
        full_df = pd.merge(full_df, self.date_info, on="date")
        full_df = pd.merge(full_df, self.store_info, on="store_id")

        full_df["year"] = full_df["date"].dt.year
        full_df["date"] = pd.to_datetime(full_df["date"])
        full_df["month"] = full_df["date"].dt.month
        full_df["day"] = full_df["date"].dt.day
        full_df = full_df[X.columns]

        cols = full_df.columns
        predictions = []

        for name, group in full_df.sort_values("date").groupby(by=["date"], group_keys=False)[cols]:
            predictions.append(self.predict_batch(group))

        predictions = np.concatenate(predictions)
        predictions[predictions < 0] = 0

        full_df = full_df.sort_values("date")
        full_df["pred"] = predictions
        full_df = full_df.sort_index()

        mask = full_df[['store_id', 'date']].apply(tuple, axis=1).isin(X[['store_id', 'date']].apply(tuple, axis=1))
        result_df = full_df[mask]

        return result_df["pred"].values

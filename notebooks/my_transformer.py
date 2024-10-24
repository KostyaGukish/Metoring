import pandas as pd
import numpy as np

from sktime.transformations.base import BaseTransformer


class MyTransformer(BaseTransformer):
    THREE_WEEKS = 21
    FIVE_WEEKS = 35
    TWO_MONTH = 61
    ONE_QUARTER = 365 // 4
    HALF_YEAR = 365 // 2
    THREE_QUARTERS = 365 * 3 // 4
    YEAR = 365

    LAGS = [
        THREE_WEEKS,
        FIVE_WEEKS,
        TWO_MONTH,
        ONE_QUARTER,
        HALF_YEAR,
        THREE_QUARTERS,
        YEAR,
    ]

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self.data = X

        if y is not None:
            self.data["visitors"] = y

        return self

    def get_store_column_lag(self, row, data, column):
        for lag in self.LAGS:
            column_name = f"store_id_{column}_{lag}days_mean"
            is_nan_column_name = f"is_nan_{column_name}"

            start_date = self.date_info["date"]
            end_date = start_date - pd.Timedelta(days=lag)

            value = data.loc[(data["date"] >= end_date), "visitors"].mean()

            if pd.isna(value):
                value = 0
                is_nan = 1
            else:
                is_nan = 0

            row[column_name] = value
            row[is_nan_column_name] = is_nan

        return row

    def get_store_features(self, row):
        store_id_data = self.data[self.data["store_id"] == row["store_id"]]
        holiday_flg_data = store_id_data[
            store_id_data["holiday_flg"] == self.date_info["holiday_flg"]
        ]
        day_of_week_data = store_id_data[
            store_id_data["day_of_week"] == self.date_info["day_of_week"]
        ]

        row = self.get_store_column_lag(row, day_of_week_data, "day_of_week")
        row = self.get_store_column_lag(row, holiday_flg_data, "holiday_flg")

        return row

    def get_area_genre_column_feature(self, group, data, column):
        for lag in self.LAGS:
            column_name = f"area_genre_{column}_{lag}days_mean"
            is_nan_column_name = f"is_nan_{column_name}"

            start_date = self.date_info["date"]
            end_date = start_date - pd.Timedelta(days=lag)

            new_data = data[data["date"] >= end_date]
            area_genre_data_mean = new_data.groupby(
                by=["date"], as_index=False
            ).visitors.mean()

            value = area_genre_data_mean.mean()["visitors"]

            if pd.isna(value):
                value = 0
                is_nan = 1
            else:
                is_nan = 0

            group[column_name] = value
            group[is_nan_column_name] = is_nan

        return group

    def get_area_genre_features(self, group, area_genre):
        area_name = area_genre[0]
        genre_name = area_genre[1]

        area_genre_data = self.data[
            (self.data["area_name"] == area_name)
            & (self.data["genre_name"] == genre_name)
        ]
        holiday_flg_data = area_genre_data[
            area_genre_data["holiday_flg"] == self.date_info["holiday_flg"]
        ]
        day_of_week_data = area_genre_data[
            area_genre_data["day_of_week"] == self.date_info["day_of_week"]
        ]

        group = self.get_area_genre_column_feature(
            group, day_of_week_data, "day_of_week"
        )
        group = self.get_area_genre_column_feature(
            group, holiday_flg_data, "holiday_flg"
        )

        return group

    def transform(self, X, y=None):
        cols = X.columns
        self.date_info = dict()
        self.date_info["date"] = X.iloc[0]["date"]
        self.date_info["day_of_week"] = X.iloc[0]["day_of_week"]
        self.date_info["holiday_flg"] = X.iloc[0]["holiday_flg"]

        for lag in self.LAGS:
            for column in ["day_of_week", "holiday_flg"]:
                for type in ["store_id", "area_genre"]:
                    column_name = f"{type}_{column}_{lag}days_mean"
                    is_nan_column_name = f"is_nan_{column_name}"

                    X.loc[:, [column_name, is_nan_column_name]] = np.nan

        X_columns = X.columns

        X = X.transform(lambda row: self.get_store_features(row), axis=1)
        X = X.groupby(by=["area_name", "genre_name"], group_keys=False)[
            X_columns
        ].apply(
            lambda group: self.get_area_genre_features(group, area_genre=group.name)
        )

        for c in X.columns:
            col_type = X[c].dtype
            if (
                col_type == "object"
                or col_type.name == "category"
                or col_type.name == "datetime64[ns]"
            ):
                X[c] = X[c].astype("category")

        return X[cols]

    def compute_rolling(self, group, column_name, lag):
        group[column_name] = (
            group[["date", "visitors"]]
            .rolling(f"{lag}D", on="date", min_periods=1)
            .mean()
            .shift()["visitors"]
        )

        return group

    def add_store_features(self, lag, column):
        column_name = f"store_id_{column}_{lag}days_mean"
        is_nan_column_name = f"is_nan_{column_name}"

        data_columns = self.data.columns
        self.data = self.data.groupby(["store_id", column], group_keys=False)[
            data_columns
        ].apply(lambda group: self.compute_rolling(group, column_name, lag))

        self.data[is_nan_column_name] = pd.isna(self.data[column_name]).astype(int)
        self.data[column_name] = self.data[column_name].fillna(0)

        return

    def add_area_genre_features(self, lag, column):
        def area_genre_compute_rolling(area_genre_data):
            area_genre_data_mean = area_genre_data.groupby(by=["date"]).visitors.mean()

            area_genre_data = (
                area_genre_data.drop(columns=["visitors"])
                .merge(area_genre_data_mean, on=["date"], how="right")
                .drop_duplicates()
            )

            area_genre_columns = area_genre_data.columns
            area_genre_data = area_genre_data.groupby(column, group_keys=False)[
                area_genre_columns
            ].apply(lambda group: self.compute_rolling(group, column_name, lag))

            return area_genre_data

        column_name = f"area_genre_{column}_{lag}days_mean"
        is_nan_column_name = f"is_nan_{column_name}"

        visitors = self.data[["visitors"]].copy()
        data_columns = self.data.columns
        self.data = self.data.groupby(["area_name", "genre_name"], group_keys=False)[
            data_columns
        ].apply(area_genre_compute_rolling)
        self.data["visitors"] = visitors["visitors"].values

        self.data[is_nan_column_name] = pd.isna(self.data[column_name]).astype(int)
        self.data[column_name] = self.data[column_name].fillna(0)

        return

    def fit_transform(self, X, y=None):
        cols = X.columns
        if self._is_fitted:
            return self.transform(X)

        else:
            self._is_fitted = True
            self.data = X

            if y is not None:
                self.data["visitors"] = y

            for lag in self.LAGS:
                for column in ["day_of_week", "holiday_flg"]:
                    self.add_area_genre_features(lag, column)
                    self.add_store_features(lag, column)

            X = self.data.drop(columns=["visitors"])

            for c in X.columns:
                col_type = X[c].dtype
                if (
                    col_type == "object"
                    or col_type.name == "category"
                    or col_type.name == "datetime64[ns]"
                ):
                    X[c] = X[c].astype("category")

            return X[cols]
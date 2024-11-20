import pandas as pd
import numpy as np

import copy

from sktime.transformations.base import BaseTransformer


class MyTransformer(BaseTransformer):
    ONE_DAY = 1
    ONE_WEEK = 7
    TWO_WEEKS = 2 * ONE_WEEK
    THREE_WEEKS = 3 * ONE_WEEK
    FIVE_WEEKS = 5 * ONE_WEEK
    TWO_MONTH = 61
    YEAR = 365
    ONE_QUARTER = YEAR // 4
    HALF_YEAR = YEAR // 2
    THREE_QUARTERS = YEAR * 3 // 4

    LAGS = [
        ONE_DAY,
        ONE_WEEK,
        TWO_WEEKS,
        YEAR,
    ]

    MEAN_LAGS = [
        # ONE_WEEK,
        TWO_WEEKS,
        THREE_WEEKS,
        FIVE_WEEKS,
        TWO_MONTH,
        # ONE_QUARTER,
        # HALF_YEAR,
        # THREE_QUARTERS,
        # YEAR,
    ]

    DAY_MAPPING = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7,
    }

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        X = copy.deepcopy(X)
        self.data = X

        if y is not None:
            self.data["visitors"] = y

        for lag in self.MEAN_LAGS:
            for column in ["day_of_week", "holiday_flg"]:
                for type in ["store_id", "area_genre"]:
                    column_name = f"{type}_{column}_{lag}days_mean"
                    is_nan_column_name = f"is_nan_{column_name}"

                    self.data.loc[:, [column_name, is_nan_column_name]] = np.nan

        self.data = self.data.sort_values("date").reset_index(drop=True)

        return self

    def to_category(self, data):
        data = copy.deepcopy(data)
        for c in data.columns:
            col_type = data[c].dtype
            if (
                col_type == "object"
                or col_type.name == "category"
                or col_type.name == "datetime64[ns]"
                or col_type.name == "string"
                or col_type == "string"
            ):
                data[c] = data[c].astype("category")

        # data = copy.deepcopy(data)
        # for c in data.columns:
        #     col_type = data[c].dtype
        #     if (
        #         col_type == "object"
        #         or col_type.name == "category"
        #         or col_type.name == "datetime64[ns]"
        #         or col_type.name == "string"
        #         or col_type == "string"
        #     ):
        #         data[c] = data[c].astype("string")

        return data

    def sin_cos(self, data):
        def column_sin_cos(df, column, period):
            df[f"sin_{column}"] = np.sin(2 * np.pi * df[column] / period)
            df[f"cos_{column}"] = np.cos(2 * np.pi * df[column] / period)
            df = data.drop(columns=column)
            return df

        data = copy.deepcopy(data)

        data["day_of_week"] = data["day_of_week"].map(self.DAY_MAPPING)

        data = column_sin_cos(data, "day_of_week", 7)
        data = column_sin_cos(data, "month", 12)
        data = column_sin_cos(data, "day", data["date"].dt.days_in_month)
        return data

    def get_store_column_lag(self, row, data, column):
        for lag in self.MEAN_LAGS:
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
        # holiday_flg_data = self.holiday_flg_data[
        #     self.holiday_flg_data["store_id"] == row["store_id"]
        # ]
        day_of_week_data = self.day_of_week_data[
            self.day_of_week_data["store_id"] == row["store_id"]
        ]

        row = self.get_store_column_lag(row, day_of_week_data, "day_of_week")
        # row = self.get_store_column_lag(row, holiday_flg_data, "holiday_flg")

        return row

    def get_area_genre_column_feature(self, group, data, column):
        for lag in self.MEAN_LAGS:
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

        holiday_flg_data = self.holiday_flg_data[
            (self.holiday_flg_data["area_name"] == area_name)
            & (self.holiday_flg_data["genre_name"] == genre_name)
        ]
        day_of_week_data = self.day_of_week_data[
            (self.day_of_week_data["area_name"] == area_name)
            & (self.day_of_week_data["genre_name"] == genre_name)
        ]

        group = self.get_area_genre_column_feature(
            group, day_of_week_data, "day_of_week"
        )
        group = self.get_area_genre_column_feature(
            group, holiday_flg_data, "holiday_flg"
        )

        return group

    def get_lag(self, X, lag):
        df = copy.deepcopy(self.data)

        df = df[["date", "visitors", "store_id"]]
        df.loc[:, "date"] = df["date"] + pd.Timedelta(days=lag)

        X = pd.merge(
            X, df, on=["date", "store_id"], how="left", suffixes=("", f"_{lag}_lag")
        )

        column_name = f"visitors_{lag}_lag"
        is_nan_column_name = f"is_nan_{column_name}"

        X = X.rename(columns={"visitors": column_name})

        X[is_nan_column_name] = pd.isna(X[column_name]).astype(int)
        X[column_name] = X[column_name].fillna(0)

        return X

    def transform(self, X, y=None):
        X = copy.deepcopy(X)

        self.date_info = dict()
        self.date_info["date"] = X.iloc[0]["date"]
        self.date_info["day_of_week"] = X.iloc[0]["day_of_week"]
        self.date_info["holiday_flg"] = X.iloc[0]["holiday_flg"]

        # for lag in self.MEAN_LAGS:
        #     for column in ["day_of_week"]: #, "holiday_flg"
        #         for type in ["store_id"]: #, "area_genre"
        #             column_name = f"{type}_{column}_{lag}days_mean"
        #             is_nan_column_name = f"is_nan_{column_name}"

        #             X.loc[:, [column_name, is_nan_column_name]] = np.nan

        # X_columns = X.columns

        self.holiday_flg_data = self.data[
            self.data["holiday_flg"] == self.date_info["holiday_flg"]
        ]
        self.day_of_week_data = self.data[
            self.data["day_of_week"] == self.date_info["day_of_week"]
        ]

        for lag in self.LAGS:
            X = self.get_lag(X, lag)

        # X = X.transform(lambda row: self.get_store_features(row), axis=1)
        # X = X.groupby(by=["area_name", "genre_name"], group_keys=False, observed=False)[
        #     X_columns
        # ].apply(
        #     lambda group: self.get_area_genre_features(group, area_genre=group.name)
        # )

        X = self.sin_cos(X)

        X = self.to_category(X)

        # cols = self.data.drop(columns=["visitors"]).columns

        return X[self.columns_order].drop(columns=["date"])

    def compute_rolling(self, group, column_name, lag):
        group[column_name] = (
            group[["date", "visitors"]]
            .rolling(window=f"{lag}D", on="date", min_periods=1)
            .mean()
            .shift()["visitors"]
        )

        return group

    def add_store_features(self, lag, column):
        column_name = f"store_id_{column}_{lag}days_mean"
        is_nan_column_name = f"is_nan_{column_name}"

        data_columns = self.data.columns
        self.data = self.data.groupby(
            ["store_id", column], group_keys=False, observed=False
        )[data_columns].apply(
            lambda group: self.compute_rolling(group, column_name, lag)
        )

        self.data[is_nan_column_name] = pd.isna(self.data[column_name]).astype(int)
        self.data[column_name] = self.data[column_name].fillna(0)

        self.data = self.data.sort_values("date").reset_index(drop=True)

        return

    def add_area_genre_features(self, lag, column):
        def area_genre_compute_rolling(area_genre_data):
            area_genre_data_mean = area_genre_data.groupby(
                by=["date"], observed=True
            ).visitors.mean()

            area_genre_data = (
                area_genre_data.drop(columns=["visitors"])
                .merge(area_genre_data_mean, on=["date"], how="right")
                .drop_duplicates()
            )

            area_genre_columns = area_genre_data.columns
            area_genre_data = area_genre_data.groupby(
                column, group_keys=False, observed=False
            )[area_genre_columns].apply(
                lambda group: self.compute_rolling(group, column_name, lag)
            )

            return area_genre_data

        column_name = f"area_genre_{column}_{lag}days_mean"
        is_nan_column_name = f"is_nan_{column_name}"

        visitors = copy.deepcopy(self.data[["visitors"]])
        data_columns = self.data.columns
        self.data = self.data.groupby(
            ["area_name", "genre_name"], group_keys=False, observed=False
        )[data_columns].apply(area_genre_compute_rolling)
        self.data["visitors"] = visitors["visitors"].values

        self.data[is_nan_column_name] = pd.isna(self.data[column_name]).astype(int)
        self.data[column_name] = self.data[column_name].fillna(0)

        self.data = self.data.sort_values("date").reset_index(drop=True)

        return

    def add_lag(self, lag):
        df = copy.deepcopy(self.data)

        df1 = df[["date", "visitors", "store_id"]]
        df1.loc[:, "date"] = df1["date"] + pd.Timedelta(days=lag)

        df = pd.merge(
            df, df1, on=["date", "store_id"], how="left", suffixes=("", f"_{lag}_lag")
        )

        column_name = f"visitors_{lag}_lag"
        is_nan_column_name = f"is_nan_{column_name}"

        df[is_nan_column_name] = pd.isna(df[column_name]).astype(int)
        df[column_name] = df[column_name].fillna(0)

        return df

    def fit_transform(self, X, y=None):
        X = copy.deepcopy(X)
        if self._is_fitted:
            return self.transform(X)

        else:
            self._is_fitted = True
            self.data = X

            if y is not None:
                self.data["visitors"] = y

            for lag in self.LAGS:
                self.data = self.add_lag(lag)

            # for lag in self.MEAN_LAGS:
            #     for column in ["day_of_week"]: #, "holiday_flg"
            #         # self.add_area_genre_features(lag, column)
            #         self.add_store_features(lag, column)

            # self.data = self.sin_cos(self.data)

            X = self.data.drop(columns=["visitors"])
            X = self.sin_cos(X)
            X = self.to_category(X)
            self.columns_order = X.columns

            return X.drop(columns=["date"])

    def update(self, X, y):
        temp = copy.deepcopy(X)
        temp["visitors"] = y
        self.data = pd.concat([self.data, temp]).reset_index(drop=True)
        self.data = self.data.drop_duplicates(subset=["store_id", "date"], keep="first")

        self.data = self.data.sort_values("date").reset_index(drop=True)

        return

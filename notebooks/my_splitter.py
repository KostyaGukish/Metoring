import pandas as pd
from numpy import np
from copy import deepcopy
from sktime.split.base import BaseWindowSplitter

TEST_SIZE = 39

class MySplitter(BaseWindowSplitter):
    def __init__(self, test_size=TEST_SIZE, fh=None, n_splits=5):
        self.test_size = test_size
        self.fh = fh
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split_data(self, data, k):
        latest_date = data["date"].max()

        start = latest_date - pd.DateOffset(days=k * self.test_size)
        end = latest_date - pd.DateOffset(days=(k - 1) * self.test_size)

        test = np.array(data[(data["date"] > start) & (data["date"] <= end)].index)
        train = np.array(data[data["date"] <= start].index)

        return train, test

    def split(self, X, y=None, groups=None):
        temp = deepcopy(X)
        temp["date"] = temp["date"].astype(str)
        temp["date"] = pd.to_datetime(temp["date"])

        for i in range(self.n_splits, 0, -1):
            train, test = self.split_data(temp, i)
            yield train, test
import unittest
import pandas as pd
import numpy as np
from my_splitter import MySplitter


class TestMySplitter(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "date": pd.date_range(start="2024-01-01", periods=100, freq="D"),
                "value": np.random.randn(100),
            }
        )
        self.splitter = MySplitter(test_size=10, n_splits=3)

        self.missing_data = pd.DataFrame(
            {
                "date": pd.date_range(start="2024-01-01", periods=100, freq="2D"),
                "value": np.random.randn(100),
            }
        )
        self.missing_splitter = MySplitter(test_size=10, n_splits=3)

    def test_get_n_splits(self):
        self.assertEqual(self.splitter.get_n_splits(), 3)
        self.assertEqual(self.missing_splitter.get_n_splits(), 3)

    def test_split(self):
        splits = list(self.splitter.split(self.data))
        self.assertEqual(len(splits), 3)

        for train, test in splits:
            self.assertTrue(np.all(np.isin(train, test) == False))
            self.assertGreater(len(test), 0)
            self.assertGreater(len(train), 0)

        splits = list(self.missing_splitter.split(self.missing_data))
        self.assertEqual(len(splits), 3)

        for train, test in splits:
            self.assertTrue(np.all(np.isin(train, test) == False))
            self.assertGreater(len(test), 0)
            self.assertGreater(len(train), 0)

    def test_split_data(self):
        train, test = self.splitter.split_data(self.data, k=1)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

        self.assertTrue(max(train) < min(test))

        train, test = self.missing_splitter.split_data(self.missing_data, k=1)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

        self.assertTrue(max(train) < min(test))


if __name__ == "__main__":
    unittest.main()

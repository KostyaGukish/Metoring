import unittest
import pandas as pd
import numpy as np
from my_transformer import MyTransformer  


class TestMyTransformer(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            "date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
            "store_id": ["1", "2", "3", "1", "2", "3", "1", "2", "3", "1"],
            "area_name": ["Area1", "Area1", "Area1", "Area1", "Area1", "Area2", "Area2", "Area2", "Area2", "Area2"],
            "genre_name": ["Genre1", "Genre1", "Genre1", "Genre1", "Genre1", "Genre2", "Genre2", "Genre2", "Genre2", "Genre2"],
            "day_of_week": ["Mon", "Mon", "Mon", "Mon", "Mon", "Mon", "Mon", "Mon", "Mon", "Mon"],
            "holiday_flg": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            "visitors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

        self.y = self.data["visitors"]
        self.transformer = MyTransformer()

    def test_fit(self):
        self.transformer.fit(self.data, self.y)

        self.assertIsNotNone(self.transformer.data, "Data should be set in fit method.")
        self.assertIn("visitors", self.transformer.data.columns, "Visitors column should be added to data in fit method.")

    def test_transform(self):
        self.transformer.fit(self.data, self.y)
        transformed_data = self.transformer.transform(self.data)

        # self.assertListEqual(list(transformed_data["area_genre_day_of_week_21days_mean"].values), [0, 1, 1.5, 2, 2.5, 0, 6, 6.5, 7, 7.5])
        self.assertIsInstance(transformed_data, pd.DataFrame, "transform should return a DataFrame.")
        self.assertFalse(transformed_data.empty, "Transformed data should not be empty.")

    def test_fit_transform(self):
        transformed_data = self.transformer.fit_transform(self.data)

        self.assertListEqual(list(transformed_data["area_genre_day_of_week_21days_mean"].values), [0, 1, 1.5, 2, 2.5, 0, 6, 6.5, 7, 7.5])
        self.assertIsInstance(transformed_data, pd.DataFrame, "fit_transform should return a DataFrame.")
        self.assertFalse(transformed_data.empty, "fit_transform data should not be empty.")
        self.assertNotIn("visitors", transformed_data.columns, "Visitors column should be dropped from fit_transform result.")

    def test_update(self):
        self.transformer.fit(self.data, self.y)
        new_data = pd.DataFrame({
            "date": pd.date_range(start="2024-01-11", periods=5, freq="D"),
            "store_id": [1, 2, 3, 1, 2],
            "area_name": ["Area1", "Area2", "Area3", "Area1", "Area2"],
            "genre_name": ["Genre1", "Genre2", "Genre3", "Genre1", "Genre2"],
            "day_of_week": ["Thu", "Fri", "Sat", "Sun", "Mon"],
            "holiday_flg": [0, 1, 0, 0, 1],
            "visitors": np.random.randint(1, 100, 5)
        })
        self.transformer.update(new_data, new_data["visitors"])

        self.assertEqual(self.data.shape[0] + new_data.shape[0], self.transformer.data.shape[0], "Data length should equals sum od lengths after update.")

    def test_to_category(self):
        converted_data = self.transformer.to_category(self.data)
        for column in ["date", "area_name", "genre_name", "day_of_week"]:
            self.assertTrue(isinstance(converted_data[column].dtype, pd.CategoricalDtype), f"{column} should be categorical.")

if __name__ == "__main__":
    unittest.main()
    # print(pd.date_range(start="2024-01-01", periods=10, freq="D"))

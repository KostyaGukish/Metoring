import unittest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from recursive_pipeline import RecursivePipeline 
from my_transformer import MyTransformer  

class TestRecursivePipeline(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({
            "date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
            "store_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            "area_name": ["Area1", "Area2", "Area3", "Area1", "Area2", "Area3", "Area1", "Area2", "Area3", "Area1"],
            "genre_name": ["Genre1", "Genre2", "Genre3", "Genre1", "Genre2", "Genre3", "Genre1", "Genre2", "Genre3", "Genre1"],
            "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"],
            "holiday_flg": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        })
        self.y = np.random.rand(10)

        self.pipeline = RecursivePipeline(
            pipeline=Pipeline(
                steps=[
                    ("transformer", MyTransformer()),
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:squaredlogerror",
                            random_state=42,
                            enable_categorical=True,
                        ),
                    ),
                ]
            )
        )

    def test_fit(self):
        self.pipeline.fit(self.X, self.y)
        self.assertTrue(hasattr(self.pipeline, 'pipeline'))

    def test_predict_batch(self):
        self.pipeline.fit(self.X, self.y)
        group = self.X.iloc[:5]  
        predictions = self.pipeline.predict_batch(group)
        self.assertEqual(predictions.shape[0], group.shape[0])

    def test_predict(self):
        self.pipeline.fit(self.X, self.y)
        predictions = self.pipeline.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.number))  

if __name__ == "__main__":
    unittest.main()

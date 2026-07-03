import unittest
from pathlib import Path

import pandas as pd

from train_model import read_dataset


class TrainModelTests(unittest.TestCase):
    def test_read_dataset_from_csv(self) -> None:
        data_path = Path("data/loan_prediction.csv")

        df = read_dataset(data_path)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("Loan_Status", df.columns)

    def test_preprocessor_uses_mean_and_mode_imputation(self) -> None:
        from train_model import build_preprocessor

        preprocessor = build_preprocessor()
        transformers = {name: transformer for name, transformer, _ in preprocessor.transformers}
        numeric_imputer = transformers["num"].named_steps["imputer"]
        categorical_imputer = transformers["cat"].named_steps["imputer"]

        self.assertEqual(numeric_imputer.strategy, "mean")
        self.assertEqual(categorical_imputer.strategy, "most_frequent")

    def test_training_pipeline_applies_smote(self) -> None:
        from train_model import SMOTE, build_training_pipeline
        from sklearn.tree import DecisionTreeClassifier

        pipeline = build_training_pipeline(DecisionTreeClassifier())
        self.assertIn("smote", pipeline.named_steps)
        self.assertIsInstance(pipeline.named_steps["smote"], SMOTE)


if __name__ == "__main__":
    unittest.main()

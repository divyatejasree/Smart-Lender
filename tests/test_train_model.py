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

    def test_preprocessor_uses_standard_scaler_for_numeric_features(self) -> None:
        from sklearn.preprocessing import StandardScaler
        from train_model import build_preprocessor

        preprocessor = build_preprocessor()
        transformers = {name: transformer for name, transformer, _ in preprocessor.transformers}
        numeric_scaler = transformers["num"].named_steps["scaler"]

        self.assertIsInstance(numeric_scaler, StandardScaler)

    def test_training_pipeline_applies_smote(self) -> None:
        from train_model import SMOTE, build_training_pipeline
        from sklearn.tree import DecisionTreeClassifier

        pipeline = build_training_pipeline(DecisionTreeClassifier())
        self.assertIn("smote", pipeline.named_steps)
        self.assertIsInstance(pipeline.named_steps["smote"], SMOTE)

    def test_preprocess_features_returns_dataframe(self) -> None:
        from train_model import preprocess_features

        data = {
            "Gender": ["Male", "Female"],
            "Married": ["Yes", "No"],
            "Dependents": ["0", "1"],
            "Education": ["Graduate", "Not Graduate"],
            "Self_Employed": ["No", "Yes"],
            "ApplicantIncome": [5000, 3000],
            "CoapplicantIncome": [0, 1500],
            "LoanAmount": [100, 120],
            "Loan_Amount_Term": [360, 360],
            "Credit_History": [1.0, 0.0],
            "Property_Area": ["Urban", "Rural"],
        }
        df = pd.DataFrame(data)

        processed = preprocess_features(df)

        self.assertIsInstance(processed, pd.DataFrame)
        self.assertEqual(processed.shape[0], df.shape[0])


if __name__ == "__main__":
    unittest.main()

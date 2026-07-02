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


if __name__ == "__main__":
    unittest.main()

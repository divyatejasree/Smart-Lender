import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.univariate_analysis import plot_univariate_analysis


class UnivariateAnalysisTests(unittest.TestCase):
    def test_plot_univariate_analysis_creates_output_files(self) -> None:
        df = pd.DataFrame(
            {
                "ApplicantIncome": [1200, 2000, 3500, 4500, 5000],
                "CoapplicantIncome": [0, 1000, 800, 600, 1200],
                "Credit_History": [0, 1, 1, 0, 1],
                "Gender": ["Male", "Female", "Male", "Male", "Female"],
                "Education": ["Graduate", "Not Graduate", "Graduate", "Graduate", "Not Graduate"],
                "Property_Area": ["Urban", "Rural", "Semiurban", "Urban", "Rural"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = plot_univariate_analysis(df, output_dir=Path(tmp_dir))

            self.assertTrue(outputs["continuous_plot"].exists())
            self.assertTrue(outputs["categorical_plot"].exists())
            self.assertEqual(
                outputs["summary"]["continuous_columns"],
                ["ApplicantIncome", "CoapplicantIncome", "Credit_History"],
            )


if __name__ == "__main__":
    unittest.main()

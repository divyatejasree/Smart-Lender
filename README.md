# Smart Lender

Smart Lender is a Flask web application that predicts whether a loan applicant is likely to be approved. It trains and compares Decision Tree, Random Forest, K-Nearest Neighbors, and XGBoost classifiers, then saves the best-performing model for real-time prediction.

If `data/loan_prediction.csv` is not available, the training script creates a realistic sample dataset so the project can run immediately.

## Project Structure

```text
Smart Lender/
  app.py
  train_model.py
  requirements.txt
  data/
  models/
  static/
    css/styles.css
  templates/
    index.html
    result.html
```

## Run Locally

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Open the app at:

```text
http://127.0.0.1:5000
```

## Dataset Columns

The app expects these applicant fields:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status

`Loan_Status` should contain `Y` for approved applicants and `N` for rejected applicants.

## Final Output

The final web output is an approval decision:

- `Loan Approved`
- `Loan Rejected`

The prediction page also displays the model confidence and selected model name.

## Entity Relationship Diagram

![Smart Lender ER Diagram](docs/er-diagram.svg)

## Documentation

- [Tools and Technologies Used](docs/tools-and-technologies.md)

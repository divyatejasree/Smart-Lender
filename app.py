from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

from train_model import MODEL_PATH, train


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)


def load_model_package() -> dict:
    if not MODEL_PATH.exists():
        train()

    try:
        return joblib.load(MODEL_PATH)
    except ModuleNotFoundError as error:
        if error.name != "xgboost":
            raise
        train()
        return joblib.load(MODEL_PATH)


def to_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def applicant_from_form(form) -> dict[str, object]:
    return {
        "Gender": form.get("gender", "Male"),
        "Married": form.get("married", "No"),
        "Dependents": form.get("dependents", "0"),
        "Education": form.get("education", "Graduate"),
        "Self_Employed": form.get("self_employed", "No"),
        "ApplicantIncome": to_float(form.get("applicant_income"), 0),
        "CoapplicantIncome": to_float(form.get("coapplicant_income"), 0),
        "LoanAmount": to_float(form.get("loan_amount"), 0),
        "Loan_Amount_Term": to_float(form.get("loan_amount_term"), 360),
        "Credit_History": to_float(form.get("credit_history"), 1),
        "Property_Area": form.get("property_area", "Urban"),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    model_package = load_model_package()
    applicant = applicant_from_form(request.form)
    input_df = pd.DataFrame([applicant])

    pipeline = model_package["pipeline"]
    prediction = int(pipeline.predict(input_df)[0])

    confidence = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_df)[0]
        confidence = round(float(probabilities[prediction]) * 100, 2)

    result = "Loan Approved" if prediction == 1 else "Loan Rejected"
    status = "approved" if prediction == 1 else "rejected"

    return render_template(
        "result.html",
        result=result,
        status=status,
        confidence=confidence,
        model_name=model_package["model_name"],
        applicant=applicant,
    )


if __name__ == "__main__":
    app.run(debug=True)

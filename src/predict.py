import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_MODEL_PATH = Path("models/lightgbm_diabetes.joblib")
DEFAULT_SCALER_PATH = Path("models/scaler.joblib")


def create_features(data: dict) -> pd.DataFrame:
    row = {
        "Pregnancies": data["Pregnancies"],
        "Glucose": data["Glucose"],
        "BloodPressure": data["BloodPressure"],
        "SkinThickness": data["SkinThickness"],
        "Insulin": data["Insulin"],
        "BMI": data["BMI"],
        "DiabetesPedigreeFunction": data["DiabetesPedigreeFunction"],
        "Age": data["Age"],
    }

    row["Insulin_BMI"] = (
        row["Insulin"] * row["BMI"]
    )

    row["Glucose_BMI"] = (
        row["Glucose"] * row["BMI"]
    )

    row["Glucose_Age"] = (
        row["Glucose"] * row["Age"]
    )

    row["BMI_Age"] = (
        row["BMI"] * row["Age"]
    )

    row["Insulin_Glucose"] = (
        row["Insulin"] * row["Glucose"]
    )

    row["Pregnancies_Age"] = (
        row["Pregnancies"] * row["Age"]
    )

    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diabetes risk prediction"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to patient JSON file",
    )

    parser.add_argument(
        "--output",
        default="results",
        help="Directory for prediction results",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Входной файл не найден: {input_path}"
        )

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:
        patient = json.load(file)

    model = joblib.load(
        DEFAULT_MODEL_PATH
    )

    scaler = joblib.load(
        DEFAULT_SCALER_PATH
    )

    features = create_features(
        patient
    )

    features_scaled = scaler.transform(
        features
    )

    prediction = int(
        model.predict(
            features_scaled
        )[0]
    )

    probability = float(
        model.predict_proba(
            features_scaled
        )[0][1]
    )

    result = {
        "prediction": prediction,
        "risk": (
            "Диабет"
            if prediction == 1
            else "Нет диабета"
        ),
        "probability": round(
            probability,
            4,
        ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "prediction.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("Результат прогнозирования:")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4,
        )
    )

    print()
    print(
        f"Результат сохранён: {output_path}"
    )


if __name__ == "__main__":
    main()